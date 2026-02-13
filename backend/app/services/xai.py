# backend/app/services/xai.py

import numpy as np
import shap
from app.services.features import pretty_feature_name, impact_sentence


class XAIEngine:
    def __init__(self, model):
        self.model = model
        self.explainer = None

        # Pipeline: preprocessador + estimador
        if hasattr(model, "steps"):
            self.preprocessor = model[:-1]
            self.estimator = model.steps[-1][1]
        else:
            self.preprocessor = None
            self.estimator = model

    def _transform(self, X):
        if self.preprocessor is None:
            Xt = X.values if hasattr(X, "values") else np.asarray(X)
            feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(Xt.shape[1])]
        else:
            Xt = self.preprocessor.transform(X)

            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()

            if hasattr(self.preprocessor, "get_feature_names_out"):
                feature_names = list(self.preprocessor.get_feature_names_out())
            else:
                feature_names = [f"f{i}" for i in range(Xt.shape[1])]

        Xt = np.asarray(Xt, dtype=np.float64)
        return Xt, feature_names

    def _ensure_explainer(self, background_X):
        if self.explainer is None:
            Xt_bg, _ = self._transform(background_X)
            self.explainer = shap.TreeExplainer(self.estimator, Xt_bg)

    @staticmethod
    def _select_positive_class(shap_values):
        if isinstance(shap_values, list):
            return shap_values[1]

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            return shap_values[:, :, 1]
        return shap_values

    def _expected_value(self):
        ev = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(ev, (list, tuple, np.ndarray)):
            if len(ev) > 1:
                return float(ev[1])
            return float(ev[0])
        return float(ev)

    def local(self, X, instance_idx=0, top_n=8, threshold=0.01):
        self._ensure_explainer(X)
        Xt, feature_names = self._transform(X)

        # ✅ FIX: desativar additivity check (evita ExplainerError)
        shap_values = self.explainer.shap_values(Xt, check_additivity=False)
        shap_values = self._select_positive_class(shap_values)

        sv = shap_values[instance_idx]

        items = [
            {
                "feature": f,
                "display_feature": pretty_feature_name(f),
                "shap_value": float(v),
                "abs_shap": float(abs(v)),
                "impact_text": impact_sentence(f, float(v)),
            }
            for f, v in zip(feature_names, sv)
        ]

        items.sort(key=lambda x: x["abs_shap"], reverse=True)
        items = items[:top_n]

        reason_codes = [it["impact_text"] for it in items if it["abs_shap"] > threshold]
        return items, reason_codes

    def global_importance(self, X, top_n=20):
        self._ensure_explainer(X)
        Xt, feature_names = self._transform(X)

        # ✅ FIX: desativar additivity check aqui também
        shap_values = self.explainer.shap_values(Xt, check_additivity=False)
        shap_values = self._select_positive_class(shap_values)

        mean_abs = np.abs(shap_values).mean(axis=0)
        std = shap_values.std(axis=0)

        items = [
            {
                "feature": f,
                "display_feature": pretty_feature_name(f),
                "mean_abs_shap": float(m),
                "std_shap": float(s),
            }
            for f, m, s in zip(feature_names, mean_abs, std)
        ]

        items.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        return items[:top_n]

    # ✅ NOVO: dados para waterfall (base + contribuições)
    def waterfall(self, X, instance_idx=0, top_n=12):
        self._ensure_explainer(X)
        Xt, feature_names = self._transform(X)

        shap_values = self.explainer.shap_values(Xt, check_additivity=False)
        shap_values = self._select_positive_class(shap_values)

        sv = shap_values[instance_idx]
        base = self._expected_value()

        contributions = [
            {
                "feature": f,
                "display_feature": pretty_feature_name(f),
                "value": float(v),
                "abs_value": float(abs(v)),
                "impact_text": impact_sentence(f, float(v)),
            }
            for f, v in zip(feature_names, sv)
        ]
        contributions.sort(key=lambda x: x["abs_value"], reverse=True)
        contributions = contributions[:top_n]

        # sinal para o gráfico
        for c in contributions:
            c["direction"] = "increases" if c["value"] > 0 else "decreases"

        return {
            "base_value": base,
            "contributions": contributions,
        }