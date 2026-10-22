import numpy as np

CONTINUOUS_FEATURES = (
    "_AGE80",
    "CHILDREN",
    "household_adults",
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "_BMI5",
    "_DRNKWEK",
    "PA1MIN_",
    "PA1VIGM_",
    "STRFREQ_",
    "FTJUDA1_",
    "FRUTDA1_",
    "BEANDAY_",
    "GRENDAY_",
    "ORNGDAY_",
    "VEGEDA1_",
)

CATEGORICAL_FEATURES = (
    "SEX",
    "_STATE",
    "_RACEGR3",
    "MARITAL",
    "EDUCA",
    "EMPLOY1",
    "INCOME2",
    "RENTHOM1",
    "VETERAN3",
    "GENHLTH",
    "BPHIGH4",
    "BPMEDS",
    "BLOODCHO",
    "TOLDHI2",
    "DIABETE3",
    "CVDSTRK3",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "_ASTHMS1",
    "HLTHPLN1",
    "PERSDOC2",
    "MEDCOST",
    "CHECKUP1",
    "FLUSHOT6",
    "PNEUVAC3",
    "QLACTLM2",
    "USEEQUIP",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
    "_SMOKER3",
    "USENOW3",
    "EXERANY2",
)


def _clean_features(x, column_names):
    columns = dict(zip(column_names, x.T))
    continuous = {
        name: columns[name].copy()
        for name in CONTINUOUS_FEATURES
        if name != "household_adults"
    }

    cellphone_adults = columns["HHADULT"].copy()
    cellphone_adults[np.isin(cellphone_adults, (77, 99))] = np.nan
    continuous["household_adults"] = np.where(
        columns["QSTVER"] < 20, columns["NUMADULT"], cellphone_adults
    )

    missing_codes = {
        "CHILDREN": (99,),
        "PHYSHLTH": (77, 99),
        "MENTHLTH": (77, 99),
        "POORHLTH": (77, 99),
        "_DRNKWEK": (99900,),
        "STRFREQ_": (99,),
    }
    for name, codes in missing_codes.items():
        values = continuous[name]
        values[np.isin(values, codes)] = np.nan

    for name in ("CHILDREN", "PHYSHLTH", "MENTHLTH", "POORHLTH"):
        values = continuous[name]
        values[values == 88] = 0

    no_unhealthy_days = (columns["PHYSHLTH"] == 88) & (columns["MENTHLTH"] == 88)
    continuous["POORHLTH"][np.isnan(columns["POORHLTH"]) & no_unhealthy_days] = 0
    for name in ("PA1MIN_", "PA1VIGM_"):
        values = continuous[name]
        values[np.isnan(values) & (columns["EXERANY2"] == 2)] = 0
    continuous["_DRNKWEK"] /= 100

    categorical = np.column_stack([columns[name] for name in CATEGORICAL_FEATURES])
    exceptions = {
        "SEX": (),
        "_STATE": (),
        "EMPLOY1": (9,),
        "INCOME2": (77, 99),
    }
    for index, name in enumerate(CATEGORICAL_FEATURES):
        values = categorical[:, index]
        codes = exceptions.get(name, (7, 9))
        values[np.isnan(values) | np.isin(values, codes)] = -1

    # -2 is a confirmed skip; -1 also covers skips caused by an unknown parent.
    categorical[
        np.isnan(columns["BPMEDS"]) & np.isin(columns["BPHIGH4"], (2, 3, 4)),
        CATEGORICAL_FEATURES.index("BPMEDS"),
    ] = -2
    categorical[
        np.isnan(columns["TOLDHI2"]) & (columns["BLOODCHO"] == 2),
        CATEGORICAL_FEATURES.index("TOLDHI2"),
    ] = -2

    return (
        np.column_stack([continuous[name] for name in CONTINUOUS_FEATURES]),
        categorical,
    )


def _encode_features(continuous, categorical, state):
    missing = np.isnan(continuous)
    filled = np.where(missing, state["medians"], continuous)
    blocks = [(filled - state["means"]) / state["scales"], missing]
    for index, categories in enumerate(state["categories"]):
        blocks.append(categorical[:, [index]] == categories[1:])
    return np.column_stack(blocks)


def fit_preprocessor(x, column_names):
    """Fit on training rows and return (processed rows, preprocessing state)."""
    continuous, categorical = _clean_features(x, column_names)
    medians = np.nanmedian(continuous, axis=0)
    filled = np.where(np.isnan(continuous), medians, continuous)
    scales = filled.std(axis=0)
    scales[scales == 0] = 1
    categories = []
    feature_names = list(CONTINUOUS_FEATURES) + [
        f"{name}_missing" for name in CONTINUOUS_FEATURES
    ]
    for name, values in zip(CATEGORICAL_FEATURES, categorical.T):
        unique = np.unique(values)
        unique = np.concatenate((unique[unique >= 0], unique[unique < 0]))
        categories.append(unique)
        for value in unique[1:]:
            label = {-1: "unknown", -2: "not_applicable"}.get(value, str(int(value)))
            feature_names.append(f"{name}={label}")
    state = {
        "medians": medians,
        "means": filled.mean(axis=0),
        "scales": scales,
        "categories": categories,
    }
    encoded = _encode_features(continuous, categorical, state)
    # These skip indicators are exact sums of their parent-question indicators.
    state["keep"] = (np.ptp(encoded, axis=0) > 0) & ~np.isin(
        feature_names, ("BPMEDS=not_applicable", "TOLDHI2=not_applicable")
    )
    state["feature_names"] = np.concatenate(
        (["intercept"], np.asarray(feature_names)[state["keep"]])
    )
    return np.column_stack((np.ones(len(x)), encoded[:, state["keep"]])), state


def transform_features(x, column_names, state):
    """Apply the training preprocessing state to validation or test rows."""
    continuous, categorical = _clean_features(x, column_names)
    encoded = _encode_features(continuous, categorical, state)
    return np.column_stack((np.ones(len(x)), encoded[:, state["keep"]]))
