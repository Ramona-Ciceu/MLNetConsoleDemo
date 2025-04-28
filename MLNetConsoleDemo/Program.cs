using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MLNetConsoleDemo;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;

class Program
{
    static void Main(string[] args)
    {
        // Step 1: Initialise MLContext
        var mlContext = new MLContext();

        // Step 2: Load data from CSV
        string dataPath = @"C:\Users\RC782\source\repos\game_data_cleaned.csv";
        var data = mlContext.Data.LoadFromTextFile<GameData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true,
            allowSparse: false
        );

        // Step 3: Split data into training and test sets (80/20)
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
        var trainSet = split.TrainSet;
        var testSet = split.TestSet;

        // Step 4: Create the pipeline
        var pipeline = mlContext.Transforms
            .ReplaceMissingValues("Dice_Roll_1", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
            .Append(mlContext.Transforms.ReplaceMissingValues("Dice_Roll_2", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Math_Operation_Encoded", nameof(GameData.Math_Operation)))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Move_Direction_Encoded", nameof(GameData.Move_Direction)))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Position_Type_Encoded", nameof(GameData.Position_Type)))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Luck_Card_Used_Encoded", nameof(GameData.Luck_Card_Used)))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Opponent_Interaction_Encoded", nameof(GameData.Opponent_Interaction)))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Decision_Taken_Encoded", nameof(GameData.Decision_Taken)))
            .Append(mlContext.Transforms.Concatenate("Features",
                nameof(GameData.Player_ID),
                nameof(GameData.Dice_Roll_1),
                nameof(GameData.Dice_Roll_2),
                nameof(GameData.Move_Value),
                nameof(GameData.Resulting_Position),
                nameof(GameData.Credits_Before),
                nameof(GameData.Credits_After),
                "Math_Operation_Encoded",
                "Move_Direction_Encoded",
                "Position_Type_Encoded",
                "Luck_Card_Used_Encoded",
                "Opponent_Interaction_Encoded",
                "Decision_Taken_Encoded"
            ))
            .Append(mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: nameof(GameData.Won_Game),
                featureColumnName: "Features",
                numberOfTrees: 100,        // Number of trees in the forest
                numberOfLeaves: 20,       // Number of leaves in each tree
                learningRate: 0.1,        // Step size for each iteration
                minimumExampleCountPerLeaf: 20  // Minimum samples per leaf to prevent overfitting
               
            ))
            .Append(mlContext.Transforms.CopyColumns("Probability", "Score"));

        // Step 5: Train the model
        Console.WriteLine("===== Training the model =====");
        var model = pipeline.Fit(trainSet);

        // Step 6: Evaluate the model
        Console.WriteLine("===== Evaluating the model =====");
        var predictions = model.Transform(testSet);
        var predictionResults = mlContext.Data.CreateEnumerable<GamePrediction>(predictions, reuseRowObject: false)
                                    .Select(pred => new
                                    {
                                        PredictedWin = pred.PredictedWin,
                                        Probability = Sigmoid(pred.Score),
                                        Score = pred.Score
                                    })
                                    .ToList();

        // Print results (no filtering by probability)
        Console.WriteLine("===== Results =====");
        foreach (var item in predictionResults)
        {
            Console.WriteLine($"PredictedWin: {item.PredictedWin}, Probability: {item.Probability:P2}, Score: {item.Score}");
        }

        // Print overall metrics (Accuracy, AUC, F1 Score)
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Won_Game");
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

        // Cross-validation results
        var cvResults = mlContext.BinaryClassification.CrossValidate(data, pipeline, numberOfFolds: 5, labelColumnName: "Won_Game");
        Console.WriteLine("\n===== Cross-Validation Results =====");
        int i = 1;
        foreach (var fold in cvResults)
        {
            Console.WriteLine($"Fold {i++}:");
            Console.WriteLine($"  Accuracy: {fold.Metrics.Accuracy:P2}");
            Console.WriteLine($"  AUC: {fold.Metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  F1 Score: {fold.Metrics.F1Score:P2}");
        }

        // Calculate average metrics across all folds
        var avgAccuracy = cvResults.Average(f => f.Metrics.Accuracy);
        var avgAUC = cvResults.Average(f => f.Metrics.AreaUnderRocCurve);
        var avgF1 = cvResults.Average(f => f.Metrics.F1Score);

        // Print average cross-validation metrics
        Console.WriteLine("\n===== Cross-Validation (Averages) =====");
        Console.WriteLine($"Avg Accuracy: {avgAccuracy:P2}");
        Console.WriteLine($"Avg AUC: {avgAUC:P2}");
        Console.WriteLine($"Avg F1 Score: {avgF1:P2}");

        // Step 7: Single Prediction
        var sample = new GameData
        {
            Player_ID = 1,
            Dice_Roll_1 = 3,
            Dice_Roll_2 = 3,
            Math_Operation = "Multiply",
            Move_Direction = "Forward",
            Move_Value = 9,
            Resulting_Position = 66,
            Position_Type = "Final_Center_Path",
            Credits_Before = 210,
            Credits_After = 210,
            Luck_Card_Used = "No",
            Opponent_Interaction = "None",
            Decision_Taken = "Moved Forward using Multiply → 9"
        };

        var samples = new List<GameData> { sample };
        var sampleDataView = mlContext.Data.LoadFromEnumerable(samples);

        var transformed = model.Transform(sampleDataView);
        var results = mlContext.Data.CreateEnumerable<GamePrediction>(transformed, reuseRowObject: false).ToList();

        // Display prediction results
        Console.WriteLine("===== Single Prediction =====");
        foreach (var result in results)
        {
            float prob = Sigmoid(result.Score);
            Console.WriteLine($"Predicted Win Probability: {prob:P2}");
            Console.WriteLine($"Confidence Threshold: {result.PredictedWin}");
        }

        // Step 8: Model Persistence
        mlContext.Model.Save(model, trainSet.Schema, "C:\\Users\\RC782\\source\\repos\\MLNetConsoleDemo\\RaceToInfinityModel.zip");
        Console.WriteLine("Press any key to exit.");
        Console.ReadKey();
    }

    // Sigmoid function to convert raw score to probability
    public static float Sigmoid(float score)
    {
        return 1 / (1 + (float)Math.Exp(-score));  // Sigmoid formula
    }
}
