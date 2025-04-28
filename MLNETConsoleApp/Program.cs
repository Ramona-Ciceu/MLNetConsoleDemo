using System;
using Microsoft.ML;
using System.Linq;
using MLNetConsoleDemo;

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Load the trained model
        var modelPath = "C:\\Users\\RC782\\source\\repos\\MLNetConsoleDemo\\RaceToInfinityModel.zip";
        var model = mlContext.Model.Load(modelPath, out _);
        var predictor = mlContext.Model.CreatePredictionEngine<GameData, GamePrediction>(model);

        // Displaying the header of the simulation
        Console.WriteLine("🎲 Race to Infinity — Smart Move Advisor 🎲");
        Console.WriteLine("Simulating Test Games...\n");

        // Define some simulated games
        var games = new[]
        {
            new { Dice_Roll_1 = 6, Dice_Roll_2 = 6, Credits = 250, Operation = "Multiply", Direction = "Forward", ExpectedOutcome = true },  // High credits, likely win
            new { Dice_Roll_1 = 3, Dice_Roll_2 = 3, Credits = 50, Operation = "Add", Direction = "Forward", ExpectedOutcome = false },     // Low credits, likely loss
            new { Dice_Roll_1 = 4, Dice_Roll_2 = 2, Credits = 150, Operation = "Subtract", Direction = "Forward", ExpectedOutcome = false }, // Medium credits, riskier scenario
        };

        // Process each game scenario
        foreach (var game in games)
        {
            var testGame = new GameData
            {
                Player_ID = 1,
                Dice_Roll_1 = 6,
                Dice_Roll_2 = 6,
                Math_Operation = "Multiply",
                Move_Direction = "Forward",
                Move_Value = 36,  // Multiplying 6 and 6
                Resulting_Position = 60,
                Position_Type = "Neutral_Space",
                Credits_Before = 200, // High starting credits
                Credits_After = 200,  // Player's credits remain the same
                Luck_Card_Used = "None",
                Opponent_Interaction = "None",
                Decision_Taken = "Move forward with Multiply",
                Won_Game = true  // Expected outcome for this test case
            };

            var prediction = predictor.Predict(testGame);
            Console.WriteLine($"Prediction: {prediction.PredictedWin}, Probability: {prediction.Probability:P2}");

            // Display the results
            Console.WriteLine($"Test Game: Dice Roll 1: {game.Dice_Roll_1}, Dice Roll 2: {game.Dice_Roll_2}, Credits Before: {game.Credits}");
            Console.WriteLine($"Operation: {game.Operation}, Direction: {game.Direction}");
            Console.WriteLine($"Expected Outcome: {(game.ExpectedOutcome ? "Win" : "Lose")}");
            Console.WriteLine($"Predicted Win: {prediction.PredictedWin}");
            Console.WriteLine($"Prediction Probability: {prediction.Probability:P2}");
            Console.WriteLine(new string('-', 40));
        }

        Console.WriteLine("Press any key to exit.");
        Console.ReadKey();
    }

    // Simple method to calculate move value based on operation
    static int CalculateMoveValue(int d1, int d2, string op)
    {
        return op switch
        {
            "Add" => d1 + d2,
            "Subtract" => Math.Abs(d1 - d2),
            "Multiply" => d1 * d2,
            "Divide" => d2 != 0 && d1 % d2 == 0 ? d1 / d2 : d1 - d2,
            _ => 0
        };
    }
}
