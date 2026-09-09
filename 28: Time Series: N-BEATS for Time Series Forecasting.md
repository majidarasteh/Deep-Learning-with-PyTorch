# N-BEATS for Time Series Forecasting — A Practical Tutorial

**N-BEATS (Neural Basis Expansion Analysis for Time Series)** is a deep learning architecture designed specifically for time-series forecasting. It is particularly interesting because it does **not require recurrent layers such as RNN, LSTM, or GRU**, and it can model complex temporal patterns using a stack of fully connected neural networks.

In this tutorial, we will build an **N-BEATS model in PyTorch** for univariate time-series forecasting, following the same general workflow we used for the previous time-series models.


# 1. What is N-BEATS?

N-BEATS stands for:

> **Neural Basis Expansion Analysis for Time Series**

The architecture was introduced by Oreshkin et al. and is based on a relatively simple idea:

> Given a historical window of observations, learn a representation of that window and use it to generate a forecast for the future horizon.

For example, suppose we have Bitcoin prices:

```text
Day 1   Day 2   Day 3   ...   Day 60
  ↓       ↓       ↓             ↓
          Historical Window
                 ↓
              N-BEATS
                 ↓
       Day 61 ... Day 67
          Forecast
```

If:

```text
window = 60
horizon = 7
```

then the model receives **60 historical observations** and predicts the **next 7 observations**.


# 2. N-BEATS vs RNN/LSTM

N-BEATS is fundamentally different from recurrent models.

### RNN/LSTM

An LSTM processes the sequence step by step:

```text
x1 → LSTM → x2 → LSTM → x3 → ... → x60
                         ↓
                      Forecast
```

The recurrent structure explicitly processes temporal information sequentially.

### N-BEATS

N-BEATS instead uses fully connected networks:

```text
Historical Window
       │
       ▼
Fully Connected Layers
       │
       ▼
Basis Expansion
       │
       ├──────────► Backcast
       │
       └──────────► Forecast
```

The model learns how the historical window can be represented using learned basis functions.


# 3. The Main Idea of N-BEATS

The key idea behind N-BEATS is **iterative residual forecasting**.

Suppose the input window is:

```text
x = [x1, x2, ..., x60]
```

The first block produces:

* a **backcast**
* a **forecast**

The backcast attempts to explain the part of the historical data that the block has learned.

The model then subtracts this backcast from the original input:

```text
Residual = Input - Backcast
```

The next block receives this residual. This is one of the most important concepts in N-BEATS.

## 3.1. N-BEATS Architecture

A simplified N-BEATS architecture looks like this:

```text
                    Input Window
                   (60 observations)
                          │
                          ▼
                 ┌─────────────────┐
                 │    Block 1      │
                 │                 │
                 │ Fully Connected │
                 │      Layers     │
                 └────────┬────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
              Backcast          Forecast
                 │                 │
                 │                 │
                 ▼                 ▼
             Residual          Forecast Sum
                 │
                 ▼
                 ┌─────────────────┐
                 │    Block 2      │
                 │                 │
                 │ Fully Connected │
                 │      Layers     │
                 └────────┬────────┘
                          │
                    ┌─────┴─────┐
                    ▼           ▼
                 Backcast    Forecast
                    │           │
                    ▼           │
                Residual         │
                    │            │
                    ▼            │
                 Block 3         │
                    │            │
                   ...           │
                    │            │
                    └────────────┘
                          │
                          ▼
                  Final Forecast
```


## 3.2. Backcast and Forecast

Each N-BEATS block produces two outputs.

## Backcast

The **backcast** represents the portion of the historical input that the block explains.

Its shape is:

```text
(batch_size, window_size)
```

For example:

```text
(batch_size, 60)
```

## Forecast

The **forecast** represents the future prediction.

Its shape is:

```text
(batch_size, horizon)
```

For example:

```text
(batch_size, 7)
```

Therefore:

```text
Block Input
      │
      ▼
   N-BEATS
      │
      ├──────────► Backcast (60 values)
      │
      └──────────► Forecast (7 values)
```


## 3.3. Residual Learning

Suppose our input is:

```text
X = [x1, x2, ..., x60]
```

The first block generates:

```text
Backcast1
Forecast1
```

We calculate:

```text
Residual1 = X - Backcast1
```

Then:

```text
Residual1 → Block 2
```

The second block produces:

```text
Backcast2
Forecast2
```

and:

```text
Residual2 = Residual1 - Backcast2
```

This continues through several blocks.

Finally, all forecasts are added:

```text
Forecast =
    Forecast1
  + Forecast2
  + Forecast3
  + ...
  + ForecastN
```

This is called **doubly residual stacking**.


## 3.4. N-BEATS Blocks

An N-BEATS model consists of multiple **blocks**.

A block contains several fully connected layers.

For example:

```text
Input
  │
  ▼
Linear
  │
ReLU
  │
  ▼
Linear
  │
ReLU
  │
  ▼
Linear
  │
ReLU
  │
  ▼
Linear
  │
  ▼
Theta
 / \
/   \
▼   ▼
Backcast  Forecast
```

The important point is that the block doesn't directly predict the future from the final hidden layer.

Instead, it generates **theta parameters**, which are then transformed into backcast and forecast representations.


## 3.5 Basis Expansion

This is another important concept in N-BEATS.

Instead of directly producing the final backcast and forecast, the network produces a vector called:

```text
θ (theta)
```

Theta is then multiplied by basis functions.

Conceptually:

```text
Neural Network
      │
      ▼
   Theta
      │
      ▼
Basis Expansion
      │
      ├────────► Backcast
      │
      └────────► Forecast
```

Mathematically, the forecast can be represented as:

$\hat{y} = \theta B $

where:

* ($\theta$) = learned coefficients
* ($B$) = basis functions
* ($\hat{y}$) = forecast

This allows N-BEATS to learn different patterns in the time series.


# 4. Preparing the Time Series

We will use the same sliding-window idea used for our previous models.

Suppose:

```text
window_size = 60
horizon = 7
```

Then the dataset creates:

```text
X₁ = observations 1–60
y₁ = observations 61–67

X₂ = observations 2–61
y₂ = observations 62–68

X₃ = observations 3–62
y₃ = observations 63–69
```

So the model learns:

```text
60 past values → 7 future values
```

## 4.1. Creating Sliding Windows

Our existing `create_sliding_windows()` function can still be used.

For example:

```python
window_size = 60
horizon = 7
```

We can create the sliding windows:

```python
X_train, y_train = create_sliding_windows(
    train_values,
    window_size=window_size,
    horizon=horizon
)

X_test, y_test = create_sliding_windows(
    test_values,
    window_size=window_size,
    horizon=horizon
)
```

The sliding-window function produces:

```text
X_train
    ↓
(number_of_samples, 60)

y_train
    ↓
(number_of_samples, 7)
```


## 4.2. Input Shape for N-BEATS

For an N-BEATS model, the input is normally represented as:

```text
(batch_size, window_size)
```

For example:

```text
(64, 60)
```

This is different from an LSTM.

For an LSTM, we typically use:

```text
(batch_size, sequence_length, features)
```

such as:

```text
(64, 60, 1)
```

For N-BEATS:

```text
(batch_size, window_size)
```

such as:

```text
(64, 60)
```

The reason is that N-BEATS uses fully connected layers rather than recurrent layers.

> **So, for our N-BEATS model, the data shape remains unchanged, just like the FFNN model.**

## 4.3. PyTorch Dataset and DataLoader

We then convert the NumPy arrays to PyTorch tensors:

```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
```

Create datasets:

We use a batch size of:

```python
batch_size = 64
```

With the `create_data_loaders()` function introduced in section 23:

```python
train_loader, test_loader = create_data_loaders(
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    y_test_tensor,
    batch_size=64
)
```

## 4.4. Implementing an N-BEATS Block

Now we can implement an N-BEATS block.

The block receives:

```text
window_size
```

and produces:

```text
backcast
forecast
```

A basic implementation is:

```python
import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        theta_size,
        backcast_size,
        forecast_size
    ):
        super().__init__()

        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.theta = nn.Linear(
            hidden_size,
            theta_size
        )

        self.backcast_linear = nn.Linear(
            theta_size,
            backcast_size
        )

        self.forecast_linear = nn.Linear(
            theta_size,
            forecast_size
        )

    def forward(self, x):

        x = self.fc_stack(x)

        theta = self.theta(x)

        backcast = self.backcast_linear(theta)

        forecast = self.forecast_linear(theta)

        return backcast, forecast
```

This is a simplified **generic N-BEATS block**.


## 4.5. Understanding the Block

The first layer receives:

```text
(batch_size, window_size)
```

For example:

```text
(64, 60)
```

The fully connected layers transform it:

```text
60
 ↓
256
 ↓
256
 ↓
256
 ↓
256
```

Then:

```text
256 → theta_size
```

Finally:

```text
theta
 ├──→ backcast
 └──→ forecast
```


## 4.6. Creating the Complete N-BEATS Model

Now we can stack multiple blocks.

```python
class NBeats(nn.Module):

    def __init__(
        self,
        window_size,
        horizon,
        hidden_size=256,
        theta_size=128,
        num_blocks=4
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=window_size,
                hidden_size=hidden_size,
                theta_size=theta_size,
                backcast_size=window_size,
                forecast_size=horizon
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):

        residual = x

        forecast = torch.zeros(
            x.size(0),
            self.blocks[0].forecast_linear.out_features,
            device=x.device
        )

        for block in self.blocks:

            backcast, block_forecast = block(residual)

            residual = residual - backcast

            forecast = forecast + block_forecast

        return forecast
```


## 4.7. Understanding the Forward Pass

Suppose we have:

```text
window_size = 60
horizon = 7
num_blocks = 4
```

The process is:

```text
Input
  │
  ▼
Block 1
  │
  ├── Backcast 1
  └── Forecast 1
  │
  ▼
Input - Backcast 1
  │
  ▼
Block 2
  │
  ├── Backcast 2
  └── Forecast 2
  │
  ▼
Residual 2
  │
  ▼
Block 3
  │
  ├── Backcast 3
  └── Forecast 3
  │
  ▼
Residual 3
  │
  ▼
Block 4
  │
  ├── Backcast 4
  └── Forecast 4
```

The final prediction is:

$\hat{y}=
\hat{y}_1+
\hat{y}_2+
\hat{y}_3+
\hat{y}_4
$


## 4.8. Why Use Multiple Blocks?

Each block can learn different aspects of the time series.

For example:

```text
Block 1
   ↓
Major pattern

Block 2
   ↓
Remaining structure

Block 3
   ↓
More complex pattern

Block 4
   ↓
Remaining residual information
```

The later blocks focus on information that earlier blocks have not already explained.

This is similar to residual learning in other deep learning architectures.


# 5. Creating the Model

For our example:

```python
window_size = 60
horizon = 7
```

we can create:

```python
model = NBeats(
    window_size=window_size,
    horizon=horizon,
    hidden_size=256,
    theta_size=128,
    num_blocks=4
)
```

We can inspect it:

```python
print(model)
```

## 5.1. Loss Function

As in our **FFNN, RNN, LSTM, and GRU** models, we use **Mean Absolute Error (MAE)**.

In PyTorch:

```python
criterion = nn.L1Loss()
```

MAE measures the average absolute difference between the actual and predicted prices.

For example:

```text
Actual price:
65000

Predicted price:
64800
```

The absolute error is:

```text
|65000 - 64800| = 200
```

The loss function calculates the average of these absolute errors over the batch.

## 5.2. Optimizer

We use the Adam optimizer:

```python
optimizer = torch.optim.Adam(
    model_N_BEATS.parameters(),
    lr=0.0001
)
```

Here:

```text
model_N_BEATS.parameters()
```

provides all trainable parameters of the N-BEATS model.

The learning rate is:

```text
0.0001
```

## 5.3. Training the Model

Because we already created a general `train_model()` function, we do not need to write another training loop specifically for `model_N_BEATS`.

We can reuse the same function:

```python
train_losses_N_BEATS, test_losses_N_BEATS = train_model(
    model=model_N_BEATS,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100
)
```

For time-series regression, the important metrics are the **loss values and forecasting errors** rather than classification accuracy.

Therefore, although our general training function may retain the accuracy variables for consistency with previous tutorials, they are not meaningful for Bitcoin price regression.


## 5.4. Visualizing the Loss

Plot Training and Test Loss

We can use the visualization function developed previously:

```python
plot_train_test_loss(
    train_losses=train_losses_N_BEATS,
    test_losses=test_losses_N_BEATS,
    title="N-BEATS Training and Test MAE",
    ylabel="MAE"
)
```

The loss curve shows how the training and test MAE change during the 100 epochs.


## 5.5 Making Predictions

Predicting the Next Bitcoin Price

Finally, we want to use the trained N-BEATS model to predict the next Bitcoin price.

Since N-BEATS receives its input in the same basic shape as our **FFNN**, we can use the `predict_next_price()` function that we introduced for the FFNN model.

We do not need to reshape the input into the 3-dimensional structure required by RNN, LSTM, GRU, or Conv1D.

We can simply use:

```python
next_price = predict_next_price(
    model=model_N_BEATS,
    data=df["Close"],
    window_size=window_size
)

print("Next Price is:", next_price.squeeze())
```

The function takes the most recent:

```text
60 Close prices
```

# 6. N-BEATS vs LSTM

It is useful to understand the difference between the two models.

| Feature                | LSTM                        | N-BEATS           |
| ---------------------- | --------------------------- | ----------------- |
| Architecture           | Recurrent                   | Fully connected   |
| Sequential processing  | Yes                         | No                |
| Hidden state           | Yes                         | No                |
| Backcast               | No                          | Yes               |
| Residual blocks        | Not fundamental             | Fundamental       |
| Multi-step forecasting | Yes                         | Yes               |
| Basis expansion        | No                          | Yes               |
| Main input shape       | `(batch, window, features)` | `(batch, window)` |


