from sklearn.base import BaseEstimator, TransformerMixin

categories = ["BEGINNER", "ADVANCED"]
subjects = ["DATASCIENCE", "FRONTEND", "BACKEND"]


def preffixed_columns(df, preffix): return [
    col for col in df.columns if col.startswith(preffix)]


class Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self


class ColumnDropper(Transformer):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X):
        data = X.copy()
        return data.drop(labels=self.columns, axis=1)


class FuzzyScaler(Transformer):
    def __init__(self, preffix):
        self.preffix = preffix

    def transform(self, X):
        df = X.copy()
        for column in preffixed_columns(df, self.preffix):
            df.loc[:, column] = df[column] / df[column].max()
        return df


class CoursesAggregator(Transformer):
    def transform(self, X):
        df = X.copy()
        df["NUM_COURSES"] = 0
        for column in preffixed_columns(df, "NUM_COURSES_"):
            df["NUM_COURSES"] += df[column]
            subj = column
            for cat in categories:
                subj = subj.replace(cat + "_", "")
            if subj not in df.columns:
                df[subj] = 0
            df[subj] += df[column]
        for cat in categories:
            for subj in subjects:
                df["PT_COURSES_" + cat + "_" + subj] = df["NUM_COURSES_" +
                                                          cat + "_" + subj] / df["NUM_COURSES_" + subj]
        return df


class SubjectMixer(Transformer):
    def __init__(self, *, subject_tuples, preffixes):
        self.subject_tuples = subject_tuples
        self.preffixes = preffixes

    def transform(self, X):
        df = X.copy()
        def substring_matcher(cols, strs): return [col for col in cols if strs]
        for preffix in self.preffixes:
            cols = preffixed_columns(df, preffix)
            for i, j in self.subject_tuples:
                subj_cols = []
                for col in cols:
                    if i in col or j in col:
                        subj_cols.append(col)
                df[preffix+i+"_"+j] = df[subj_cols].sum(axis=1)
        return df


class ScorePenalty(Transformer):
    def __init__(self, *, penalty, ref_preffix="PT_COURSES_BEGINNER_", preffix="AVG_SCORE_"):
        self.penalty = penalty
        self.preffix = preffix
        self.ref_preffix = ref_preffix

    def transform(self, X):
        df = X.copy()
        for col in preffixed_columns(df, self.preffix):
            subj = col.replace(self.preffix, "")
            df["PENALIZED_" + col] = df[col] * \
                (1 - df[self.ref_preffix + subj]*self.penalty)
        return df
