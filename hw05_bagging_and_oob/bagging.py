import numpy as np


class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        """
        Generate indices for every bag and store in self.indices_list list
        """
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            arr = np.arange(data_length)
            idxs = np.random.choice(arr, size=data_length)
            self.indices_list.append(idxs)

    def fit(self, model_constructor, data, target):
        """
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        """
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert (
            len(set(list(map(len, self.indices_list)))) == 1
        ), "All bags should be of the same length!"
        assert list(map(len, self.indices_list))[0] == len(
            data
        ), "All bags should contain `len(data)` number of elements!"
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            indxs = self.indices_list[bag]
            data_bag, target_bag = data[indxs], target[indxs]  # Your Code Here
            self.models_list.append(
                model.fit(data_bag, target_bag)
            )  # store fitted models here
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        """
        Get average prediction for every object from passed dataset
        """
        # Your code here
        arr_predicts = np.zeros(data.shape[0])
        for model in self.models_list:
            arr_predicts += model.predict(data)

        arr_predicts /= len(self.models_list)

        return arr_predicts

    def _get_oob_predictions_from_every_model(self):
        """
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        """
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        for i, data_i in enumerate(self.data):
            for k, indxs in enumerate(self.indices_list):
                if i not in indxs:
                    prediction = self.models_list[k].predict(data_i.reshape(1, -1))
                    list_of_predictions_lists[i].append(prediction)

        self.list_of_predictions_lists = np.array(
            list_of_predictions_lists, dtype=object
        )

    def _get_averaged_oob_predictions(self):
        """
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        """
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = []
        # Your Code Here
        for predicts_list in self.list_of_predictions_lists:
            if len(predicts_list) > 0:
                self.oob_predictions.append(np.mean(predicts_list))
            else:
                self.oob_predictions.append(None)

    def OOB_score(self):
        """
        Compute mean square error for all objects, which have at least one prediction
        """
        self._get_averaged_oob_predictions()
        Loss = 0
        number_not_None = 0
        for predict, t in zip(self.oob_predictions, self.target):
            if predict is not None:
                Loss += (predict - t) ** 2
                number_not_None += 1

        Loss /= number_not_None
        return Loss
