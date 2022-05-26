from carla import MLModelCatalog
from carla.models.catalog.load_model import save_model
from carla.models.catalog.train_model import DataFrameDataset, _training_torch
from torch.utils.data import DataLoader


class CustomMLModel(MLModelCatalog):
    def retrain(
        self,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        hidden_size=[18, 9, 3],
    ):
        """
        Parameters
        ----------
        learning_rate: float
            Learning rate for the training.
        epochs: int
            Number of epochs to train for.
        batch_size: int
            Number of samples in each batch
        force_train: bool
            Force training, even if model already exists in cache.
        hidden_size: list[int]
            hidden_size[i] contains the number of nodes in layer [i]
        n_estimators: int
            Number of estimators in forest.
        max_depth: int
            Max depth of trees in the forest.
        Returns
        -------
        """
        layer_string = "_".join([str(size) for size in hidden_size])
        if self.model_type == "linear" or self.model_type == "forest":
            save_name = f"{self.model_type}"
        elif self.model_type == "ann":
            save_name = f"{self.model_type}_layers_{layer_string}"
        else:
            raise NotImplementedError("Model type not supported:", self.model_type)

        # get preprocessed data
        df_train = self.data.df_train
        df_test = self.data.df_test

        x_train = df_train[list(set(df_train.columns) - {self.data.target})]
        y_train = df_train[self.data.target]
        x_test = df_test[list(set(df_test.columns) - {self.data.target})]
        y_test = df_test[self.data.target]

        # order data (column-wise) before training
        x_train = self.get_ordered_features(x_train)
        x_test = self.get_ordered_features(x_test)

        train_dataset = DataFrameDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = DataFrameDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        _training_torch(
            self._model,
            train_loader,
            test_loader,
            learning_rate,
            epochs,
        )

        save_model(
            model=self._model,
            save_name=save_name,
            data_name=self.data.name,
            backend=self.backend,
        )