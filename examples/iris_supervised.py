from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from spilearn.estimators import (
    ReceptiveFieldsTransformer,
    SupervisedTemporalClassifier,
)


def build_pipeline():
    settings = {"topology": {}, "network": {}, "learning": {}}
    model = {
        "neuron_out_model": "iaf_psc_exp",
        "neuron_out": {
            "C_m": 1.0,
            "E_L": 0.0,
            "I_e": 0.0,
            "V_m": 0.0,
            "V_reset": 0.0,
            "V_th": 5.0,
            "tau_m": 20.0,
            "tau_minus": 25.0,
            "t_ref": 100.0,
        },
        "syn_dict_exc": {
            "weight": {
                "type": "uniform",
                "specs": {"min": 0.0, "max": 1.0},
            },
            "mu_plus": 0.0,
            "mu_minus": 0.0,
            "lambda": 0.03,
            "tau_plus": 6.0,
            "alpha": 0.65,
            "Wmax": 1.0,
            "synapse_model": "stdp_synapse",
        },
        "syn_dict_inh": {
            "weight": -10.0,
            "synapse_model": "static_synapse",
        },
    }

    return make_pipeline(
        MinMaxScaler(),
        ReceptiveFieldsTransformer(n_fields=10, scale=3, reshape=True),
        SupervisedTemporalClassifier(
            settings,
            model,
            n_input=40,
            n_layer_out=3,
            epochs=1,
            h_time=150,
            start_delta=50,
            h=0.01,
            teacher_amplitude=700,
            num_threads=1,
            use_inhibition=True,
            test_with_inhibition=False,
            threshold=5,
        ),
    )


def main():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print(f"accuracy={accuracy_score(y_test, pipeline.predict(X_test)):.3f}")


if __name__ == "__main__":
    main()
