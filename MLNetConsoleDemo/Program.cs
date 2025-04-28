using Microsoft.ML;
using Microsoft.ML.Data;
using MLNetConsoleDemo;
using System;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Load turn data
        string dataPath = @"C:\Users\RC782\source\repos\turn_data.csv"; // <-- make sure this path points to your turn_data.csv
        var data = mlContext.Data.LoadFromTextFile<GameData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true
        );

        // Split into Train/Test
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
        var trainSet = split.TrainSet;
        var testSet = split.TestSet;

        // Build the training pipeline
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(GameData.Best_Move_Operation))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Luck_Card_Used_Encoded", nameof(GameData.Luck_Card_Used)))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("In_Lock_Encoded", nameof(GameData.In_Lock)))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Opponent_Close_Encoded", nameof(GameData.Opponent_Close)))
            .Append(mlContext.Transforms.Concatenate("Features",
                nameof(GameData.Dice_Roll_1),
                nameof(GameData.Dice_Roll_2),
                nameof(GameData.Credits_Before),
                nameof(GameData.Position_Before),
                "Luck_Card_Used_Encoded",
                "In_Lock_Encoded",
                "Opponent_Close_Encoded"
            ))
            .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                labelColumnName: "Label",
                featureColumnName: "Features"
            ))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")); // map back to text label

        // Train
        Console.WriteLine("===== Training Model =====");
        var model = pipeline.Fit(trainSet);

        // Evaluate
        Console.WriteLine("===== Evaluating Model =====");
        var predictions = model.Transform(testSet);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");
        Console.WriteLine($"LogLoss: {metrics.LogLoss:F4}");

        // Save model
        mlContext.Model.Save(model, trainSet.Schema, @"C:\Users\RC782\source\repos\RaceToInfinity_MovePredictor.zip");

        Console.WriteLine("Model training and saving complete! Press any key to EXIT");
        Console.ReadKey();
    }
}
