import torch
import torch.nn as nn
import torch.optim as optim

from assignment import generate_dataset, N_TRIALS_TRAIN
from myRNN import TaskRNN


# --------------------
# 1. Generate training data
# --------------------

X_train, Y_train = generate_dataset(N_TRIALS_TRAIN)

X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train)


# --------------------
# 2. Train 5 models
# --------------------

N_MODELS = 5
N_EPOCHS = 100


for model_idx in range(N_MODELS):

    print(f"\n===== Model {model_idx + 1} =====")

    # Create a new RNN
    model = TaskRNN(
        input_size=12,
        hidden_size=64,
        output_size=5
    )


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )


    # --------------------
    # 3. Training
    # --------------------

    for epoch in range(N_EPOCHS):

        model.train()

        optimizer.zero_grad()

        output, hidden_activity = model(X_train)

        loss = criterion(output, Y_train)

        loss.backward()

        optimizer.step()


        predicted = torch.argmax(
            output,
            dim=1
        )

        accuracy = (
            predicted == Y_train
        ).float().mean()


        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch + 1:3d} | "
                f"Loss: {loss.item():.4f} | "
                f"Accuracy: {accuracy.item():.4f}"
            )


    # --------------------
    # 4. Save model
    # --------------------

    torch.save(
        model.state_dict(),
        f"task2_rnn_{model_idx + 1}.pth"
    )

    print(
        f"Model {model_idx + 1} saved."
    )


print("\nAll 5 models finished.")