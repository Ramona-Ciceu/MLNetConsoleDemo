
 RaceToInfinity - Machine Learning Move Predictor

RaceToInfinity is a machine-learning-based decision support system that predicts optimal player moves in a board game setting. The project leverages ML.NET to train a model using game data, evaluate its performance, and simulate game decisions. It includes a training pipeline, console-based simulation, and unit testing setup to validate the model's performance.



  Project Structure



RaceToInfinity/
├── MLNetConsoleDemo/         # Project that trains the ML model
│   ├── GameData.cs
│   ├── GamePrediction.cs
│   ├── Program.cs            # Training logic
│   └── RaceToInfinity.zip    # Trained model (saved to project root)
│
├── MLNETConsoleApp/          # Console app to run move simulations
│   └── Program.cs            # Loads model and simulates game outcomes
│
├── RaceToInfinityTests/      # Unit tests for model and data validation
│   ├── DataTests.cs
│   ├── ModelTests.cs
│   ├── turn\_data.csv         # Test data for training/validation
│   └── RaceToInfinity.zip    # Copied model for testing use



 Features

-  ML.NET Training Pipeline** using `LbfgsMaximumEntropy` for multiclass classification
-  Prediction Engine**: Suggests the best move based on dice rolls, board position, and game state
-  Model Persistence**: Saves the trained model as `RaceToInfinity.zip` to the current project folder
-  Unit Tests** using xUnit to verify:
  - Valid CSV column structure
  - Valid predicted moves from the model
  - Safe and portable model loading
-  Simulation Console App**: Loads the model and simulates multiple move predictions
-  Dynamic Pathing**: All paths to data and model files are dynamically built using `Directory.GetCurrentDirectory()` and relative traversal to ensure portability and GitHub compatibility



  Testing

To support testing:
- The files `turn_data.csv` and `RaceToInfinity.zip` were placed in the `RaceToInfinityTests` folder.
- The model is loaded in `ModelTests.cs` using a dynamic path that navigates from `bin/Debug/netX.X` to the project root.

 This ensures tests can be executed regardless of the environment.



  Simulation Game Execution

- After training the model in `MLNetConsoleDemo`, the `RaceToInfinity.zip` file is manually copied to the `MLNETConsoleApp` project folder.
- This enables the `MLNETConsoleApp` to simulate gameplay and predict moves using the trained model.

 The simulation logic dynamically loads the model from the current project directory using `AppDomain.CurrentDomain.BaseDirectory` or equivalent traversal logic.



 Data Features Used for Prediction

- Dice_Roll_1 and Dice_Roll_2
- Credits_Before
- Position_Before
- Luck_Card_Used
- In_Lock
- Opponent_Close
- Move_Direction

 Target label: `Best_Move_Operation` (Add, Subtract, Multiply, Divide)



 Requirements

- .NET 7.0 or later
- Visual Studio or VS Code
- ML.NET NuGet packages
- xUnit for testing



Supervisor Communication

Project updates and collaboration were maintained through regular emails and scheduled meetings with the supervisor. All resources have been organized clearly to ensure reproducibility and transparency.



 License

This project is part of an academic submission and may be reused or extended for educational purposes.

