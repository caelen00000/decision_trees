import numpy
import pandas
from treelib import Node, Tree
import matplotlib.pyplot as pyplot
import sklearn.linear_model
import sklearn.metrics
import graphviz

rng = numpy.random.default_rng(seed=0)

def get_codon_usage_df():
    # Obtained from Kaggle: https://www.kaggle.com/datasets/meetnagadia/condon-usage-dataset
    # Original data from this paper: https://www.nature.com/articles/s41598-023-28965-7

    # deleted SpeciesID 12440 and 353569 in codon_usage_fixed.csv. It looks like they left punctuation in SpeciesName which resulted in missing codon data.
    codon_usage = pandas.read_csv("codon_usage_fixed.csv")

    del codon_usage["Ncodons"] # Total number of codons before normalization
    del codon_usage["SpeciesID"] # NCBI species IDs


    """
    #TODO: for testing with largest classes
    codon_usage = codon_usage.loc[(codon_usage["Kingdom"] == "bct")
                                  | (codon_usage["Kingdom"] == "vrl")
                                  | (codon_usage["Kingdom"] == "pln")
                                  | (codon_usage["Kingdom"] == "inv")
                                  | (codon_usage["Kingdom"] == "vrt")]"""

    codon_usage["Kingdom"] = codon_usage["Kingdom"].astype('category')
    codon_usage["SpeciesName"] = codon_usage["SpeciesName"].astype('category')

    # from the Kaggle page
    dna_type_map = {
        0: "genomic",
        1: "mitochondrial",
        2: "chloroplast",
        3: "cyanelle",
        4: "plastid",
        5: "nucleomorph",
        6: "secondary_endosymbiont",
        7: "chromoplast",
        8: "leucoplast",
        9: "NA",
        10: "proplastid",
        11: "apicoplast",
        12: "kinetoplast"
    }

    codon_usage["DNAtype"] = codon_usage["DNAtype"].replace(dna_type_map)
    codon_usage["DNAtype"] = codon_usage["DNAtype"].astype('category')

    return codon_usage

def get_accuracy_from_prediction(prediction, y):
    return sum(prediction == y) / len(y)

def get_max_gain_ratio_split(full_df,
                             sorted_df,
                             label_dummies,
                             min_split_size = 2):

    n = sorted_df.shape[0]

    #if n < 2 * min_split_size:
    #    return full_df.iloc[:0], full_df.iloc[:0], 0.0, 0.0

    label_matrix = label_dummies.loc[sorted_df.index].to_numpy(dtype=numpy.int32, copy=False)

    label_totals = label_matrix.sum(axis=0)
    p_total = label_totals / n
    info_pre_split = numpy.sum(-p_total * numpy.log2(p_total,
                                                     out=numpy.zeros_like(p_total, dtype=numpy.float64),
                                                     where=(p_total != 0)))

    lower_counts = numpy.cumsum(label_matrix[:-1], axis=0)
    lower_n = numpy.arange(1, n, dtype=numpy.int32)
    upper_n = n - lower_n
    upper_counts = label_totals - lower_counts

    lower_probs = numpy.divide(lower_counts,
                               lower_n[:, None],
                               out=numpy.zeros_like(lower_counts, dtype=numpy.float64),
                               where=(lower_counts != 0))
    upper_probs = numpy.divide(upper_counts,
                               upper_n[:, None],
                               out=numpy.zeros_like(upper_counts, dtype=numpy.float64),
                               where=(upper_counts != 0))

    info_lower = -numpy.sum(lower_probs * numpy.log2(lower_probs,
                                                      out=numpy.zeros_like(lower_probs),
                                                      where=(lower_probs != 0)),
                            axis=1)
    info_upper = -numpy.sum(upper_probs * numpy.log2(upper_probs,
                                                      out=numpy.zeros_like(upper_probs),
                                                      where=(upper_probs != 0)),
                            axis=1)

    info_x = lower_n / n * info_lower + upper_n / n * info_upper
    gain = info_pre_split - info_x

    p_lower = lower_n / n
    p_upper = upper_n / n
    split_info = -(p_lower * numpy.log2(p_lower)) - (p_upper * numpy.log2(p_upper))

    gain_ratio = numpy.divide(gain,
                              split_info,
                              out=numpy.zeros_like(gain, dtype=numpy.float64),
                              where=(split_info != 0))

    feature_values = sorted_df["feature"].to_numpy()
    valid_cuts = ((lower_n >= min_split_size) &
                  (upper_n >= min_split_size) &
                  (feature_values[1:] > feature_values[:-1]))

    #if not numpy.any(valid_cuts):
    #    return full_df.iloc[:0], full_df.iloc[:0], 0.0, 0.0

    candidate_gain_ratio = numpy.where(valid_cuts, gain_ratio, -numpy.inf)
    max_gain_ratio_i = int(numpy.argmax(candidate_gain_ratio))
    max_gain_ratio = float(candidate_gain_ratio[max_gain_ratio_i])

    lower_split_feat_max = feature_values[max_gain_ratio_i]
    upper_split_feat_min = feature_values[max_gain_ratio_i + 1]

    split_threshold = (lower_split_feat_max + upper_split_feat_min) / 2

    lower_split = full_df.loc[sorted_df.index[:max_gain_ratio_i + 1]]
    upper_split = full_df.loc[sorted_df.index[max_gain_ratio_i + 1:]]


    return lower_split, upper_split, split_threshold, max_gain_ratio

def split_df(df, feature_sorted_dict, label_col):
    max_gain_ratio = 0
    max_gain_ratio_feature = ""
    max_gain_ratio_lower_split = None
    max_gain_ratio_upper_split = None
    max_gain_ratio_split_threshold = 0

    label_dummies = pandas.get_dummies(df[label_col], dtype=numpy.int32)

    for feature, sorted_df in feature_sorted_dict.items():
        lower_split, upper_split, split_threshold, gain_ratio = get_max_gain_ratio_split(df,
                                                                                         sorted_df,
                                                                                         label_dummies)

        if gain_ratio > max_gain_ratio:
            max_gain_ratio = gain_ratio
            max_gain_ratio_feature = feature
            max_gain_ratio_lower_split = lower_split
            max_gain_ratio_upper_split = upper_split
            max_gain_ratio_split_threshold = split_threshold

    feature_sorted_dict_lower_split = {}
    feature_sorted_dict_upper_split = {}

    lower_split_index = max_gain_ratio_lower_split.index

    for feature, sorted_df in feature_sorted_dict.items():
        lower_mask = sorted_df.index.isin(lower_split_index)
        feature_sorted_dict_lower_split[feature] = sorted_df[lower_mask]
        feature_sorted_dict_upper_split[feature] = sorted_df[~lower_mask]

    return (max_gain_ratio_lower_split,
            max_gain_ratio_upper_split,
            feature_sorted_dict_lower_split,
            feature_sorted_dict_upper_split,
            max_gain_ratio_split_threshold,
            max_gain_ratio_feature,
            max_gain_ratio)

class Classifier:
    def __init__(self, train_df, test_df, label_col = "Kingdom", first_feature_col_index = 2):
        self.train_df = train_df
        self.test_df = test_df

        self.label_col = label_col

        self.n_train = train_df.shape[0]
        self.n_test = test_df.shape[0]

        self.x_train = train_df.iloc[:, first_feature_col_index:].to_numpy()
        self.x_test = test_df.iloc[:, first_feature_col_index:].to_numpy()

        self.y_train = train_df[label_col].cat.codes.to_numpy()
        self.y_test = test_df[label_col].cat.codes.to_numpy()

        self.n_classes = train_df[label_col].cat.codes.nunique()
        self.n_features = train_df.shape[1] - first_feature_col_index

class TreeData:
    def __init__(self, df, feature_sorted_dict, test_attribute = None, test_cutoff = None, is_leaf = False, label_col = "Kingdom"):
        self.df = df
        self.feature_sorted_dict = feature_sorted_dict
        self.cases = df.shape[0]
        self.classes = df[label_col].unique()
        self.label_col = label_col

        self.test_attribute = test_attribute
        self.test_cutoff = test_cutoff
        self.gain_ratio = None
        self.is_leaf = is_leaf

class DecisionTree(Classifier):
    def __init__(self,
                 train_df,
                 test_df,
                 label_col="Kingdom",
                 first_feature_col_index=3,
                 max_depth: int | None = None):

        super().__init__(train_df, test_df, label_col, first_feature_col_index)

        feature_columns = train_df.iloc[:, first_feature_col_index:].columns

        _feature_sorted_dict = {}

        for feat in feature_columns:
            _feature_sorted_dict[feat] = train_df.loc[:, [label_col, feat]].sort_values(feat).rename(columns={feat: "feature"})

        self.max_depth = max_depth

        self.tree_id = "tree"
        self.tree = Tree(identifier=self.tree_id)

        self.tree.create_node(identifier="root",
                              data=TreeData(self.train_df, _feature_sorted_dict))

    def train(self):
        def build_tree(subtree_root):
            if self.max_depth is not None and self.tree.depth(self.tree[subtree_root]) == self.max_depth:
                return

            current_treedata = self.tree[subtree_root].data

            #if current_treedata.cases == 0:
            if current_treedata.cases < 4:
                current_treedata.is_leaf = True

                #TODO: mode() assumes classes is a series, and there can be multiple modes, so the first is taken
                #current_treedata.classes = self.tree[self.tree[subtree_root].predecessor(self.tree_id)].data.classes.mode()[0]
            else:
                if len(current_treedata.classes) == 1:
                    current_treedata.is_leaf = True
                else:
                    print(current_treedata.df.shape)

                    max_gain_ratio_lower_split, max_gain_ratio_upper_split, feature_sorted_dict_lower_split, feature_sorted_dict_upper_split, max_gain_ratio_split_threshold, max_gain_ratio_feature, max_gain_ratio = split_df(
                        current_treedata.df, current_treedata.feature_sorted_dict, self.label_col)

                    self.tree[subtree_root].data.test_attribute = max_gain_ratio_feature
                    self.tree[subtree_root].data.test_cutoff = max_gain_ratio_split_threshold
                    self.tree[subtree_root].data.gain_ratio = max_gain_ratio

                    print(max_gain_ratio_feature + " " + str(max_gain_ratio))

                    lower_split_id = max_gain_ratio_feature + " <= " + str(max_gain_ratio_split_threshold)
                    build_tree(self.tree.create_node(tag=lower_split_id,
                                          data=TreeData(max_gain_ratio_lower_split, feature_sorted_dict_lower_split),
                                          parent=subtree_root).identifier)

                    upper_split_id = max_gain_ratio_feature + " > " + str(max_gain_ratio_split_threshold)
                    build_tree(self.tree.create_node(tag=upper_split_id,
                                          data=TreeData(max_gain_ratio_upper_split, feature_sorted_dict_upper_split),
                                          parent=subtree_root).identifier)

        build_tree(self.tree.root)

class Perceptron(Classifier):
    def __init__(self, train_df, test_df, max_iter, learning_rate: float = 1, label_col = "Kingdom", first_feature_col_index = 3):
        train_df.insert(loc=first_feature_col_index, column="bias_feature", value=1.0)
        test_df.insert(loc=first_feature_col_index, column="bias_feature", value=1.0)

        super().__init__(train_df, test_df, label_col, first_feature_col_index)

        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.w = numpy.zeros((self.n_classes, self.n_features))

        self.history = {
            "epoch": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "train_accuracy_max": 0.0,
            "test_accuracy_max": 0.0,
            #"train_accuracy_per_class": [],
            #"test_accuracy_per_class": [],
        }

    def classify(self, x):
        return numpy.argmax(numpy.vecmat(self.w, x.T), axis = 0)

    def get_accuracy(self, x, y):
        return get_accuracy_from_prediction(self.classify(x), y)

    def train(self, test_interval = 100):
        max_accuracy = 0
        pocket_weights = self.w.copy()

        for epoch in range(self.max_iter):
            for sample in rng.permutation(self.n_train):
                predicted = self.classify(self.x_train)

                sample_predicted = predicted[sample]
                sample_true = self.y_train[sample]

                if sample_predicted != sample_true:
                    self.w[sample_true] = self.w[sample_true] + self.learning_rate * self.x_train[sample]

                    self.w[sample_predicted] = self.w[sample_predicted] - self.learning_rate * self.x_train[sample]

                    break

            # reclassify on updated weights
            predicted = self.classify(self.x_train)

            accuracy = get_accuracy_from_prediction(predicted, self.y_train)

            if epoch % test_interval == 0:
                self.history["epoch"].append(epoch)
                self.history["train_accuracy"].append(accuracy)

                test_accuracy = self.get_accuracy(self.x_test, self.y_test)
                self.history["test_accuracy"].append(test_accuracy)

                if accuracy > self.history["train_accuracy_max"]:
                    self.history["train_accuracy_max"] = accuracy

                if test_accuracy > self.history["test_accuracy_max"]:
                    self.history["test_accuracy_max"] = test_accuracy

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                pocket_weights = self.w.copy()

        self.w = pocket_weights










class RandomForest(Classifier):
    pass

def subsample(df, keep_proportion):
    return df.loc[df.index[0:int(len(df.index) * keep_proportion)]]

def train_test_split(df: pandas.DataFrame, test_proportion, label_col= "Kingdom"):
    labels = df[label_col].cat.codes
    
    test_indices = rng.choice(labels.index,
                              size=int(len(labels.index) * test_proportion),
                              replace=False,
                              shuffle=True)  # TODO: relying on shuffle to hopefully run into rare classes during training, need to implement stratified or remove rare classes

    train_df = df.drop(index=test_indices)
    test_df = df.loc[test_indices]
    
    return train_df, test_df

if __name__ == "__main__":
    codon_usage = get_codon_usage_df()

    # TODO: perhaps keep mitochondrial?
    # Keeping everything greatly reduces performance
    #codon_usage = codon_usage.loc[codon_usage["DNAtype"] == "genomic"]
    #del codon_usage["DNAtype"]

    # full dataset
    #my_train_df, my_test_df = train_test_split(codon_usage, test_proportion=0.1)

    # subsample for testing
    my_train_df, my_test_df = train_test_split(subsample(codon_usage, keep_proportion = 1), test_proportion=0.1)

    print(my_train_df.shape)
    print(my_test_df.shape)

    tr = DecisionTree(my_train_df, my_test_df, max_depth=10)
    tr.train()

    tree_file_prefix = "tree"
    tr.tree.to_graphviz(tree_file_prefix + ".dot")

    graphviz_graph = graphviz.Source.from_file(tree_file_prefix + ".dot")
    graphviz_graph.render(tree_file_prefix, format="png", cleanup=True)

    """
    my_test_df = codon_usage.loc[test_indices]
    my_train_df = codon_usage.drop(index=test_indices)

    perc = Perceptron(my_train_df, my_test_df, max_iter=20000, learning_rate = 1) # Apparently perceptron LR only scales the weights?

    perc.train(test_interval=1)

    print(perc.w)
    print(perc.get_accuracy(perc.x_train, perc.y_train))
    #print(perc.get_accuracy(perc.x_test, perc.y_test))

    pyplot.plot(perc.history["epoch"], perc.history["train_accuracy"])
    #pyplot.plot(perc.history["epoch"], perc.history["test_accuracy"])
    pyplot.ylim([0, 1])
    pyplot.legend(["Training Accuracy", "test"])

    pyplot.savefig("perceptron.png")

    pyplot.show()



    print(perc.history["train_accuracy_max"])
    print(perc.history["test_accuracy_max"])



    #print(perc.x_train)
    #print(perc.x_test)

    #print(my_test_df)
    #print(my_train_df)

    

    perceptron2 = sklearn.linear_model.Perceptron()
    perceptron2.fit(perc.x_train, perc.y_train)

    print(f'Accuracy: {sklearn.metrics.accuracy_score(perc.y_test, perceptron2.predict(perc.x_test))}')
    """


