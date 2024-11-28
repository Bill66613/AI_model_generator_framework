from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed

class ModelTrainer:
    def __init__(self):
        self.model = None

    def train_linear_regression(self, X_train, y_train):
        """Train a linear regression model."""
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def train_decision_tree(self, X_train, y_train):
        """Train a decision tree model."""
        self.model = DecisionTreeRegressor(random_state=42)
        self.model.fit(X_train, y_train)

    def train_random_forest(self, X_train, y_train):
        """Train a random forest model."""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def train_neural_network(self, X_train, y_train, epochs=10):
        """Train a simple neural network."""
        self.model = Sequential([
            Dense(64, activation='relu', input_dim=X_train.shape[1]),
            Dense(32, activation='relu'),
            Dense(1)  # Single output for regression
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    # def train_rnn(self, X_train, y_train, epochs=10):
    #     """Train an RNN for sequential data."""
    #     self.model = Sequential([
    #         LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    #         Dense(1)  # Single output for regression
    #     ])
    #     self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #     self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    def train_rnn(self, X_train, y_train, epochs=10, batch_size=32):
        """Train an RNN for sequential data."""
        self.model = Sequential([
            LSTM(32, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            Dense(1, activation='linear')  # Adjust based on regression or classification
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # def train_lstm(self, X_train, y_train, epochs=10):
    #     """Train an LSTM for sequential data."""
    #     self.model = Sequential([
    #         LSTM(128, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    #         LSTM(64, activation='relu'),
    #         Dense(1)  # Single output for regression
    #     ])
    #     self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #     self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

    def train_lstm(self, X_train, y_train, epochs=10, batch_size=32):
        """Train an LSTM for sequential data."""
        self.model = Sequential([
            LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            Dense(1, activation='linear')  # Adjust based on regression or classification
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        if isinstance(self.model, Sequential):
            # For Keras models
            loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
            return {"loss": loss, "mae": mae}
        else:
            # For sklearn models
            score = self.model.score(X_test, y_test)
            return {"r2_score": score}
