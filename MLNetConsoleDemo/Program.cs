using Microsoft.ML;
using Microsoft.ML.Data;
using MLNetConsoleDemo;
using System;
using System.IO;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        /* **********************************************
           * Step 1: Initialise MLContext
           * MLContext is the core object in ML.NET
           * It provides functions for data loading, model training, and evaluation.
         ********************************************** */
        var mlContext = new MLContext();


        /* ************************************************************
             * Step 2: Load Data from csv
             * Loads a CSV file (turn_data.csv) into an IDataView (ML.NET's data format).
             * Uses the GameData class (not shown in your code but must be defined elsewhere).
             * hasHeader: true → The first row contains column names.
             * separatorChar: ',' → Uses a comma as the delimiter.
          ********************************************************************** */
        // Get current working directory (e.g., bin/Debug/net7.0)
        string currentDir = Directory.GetCurrentDirectory();
        // Navigate to the project root (up 3 levels)
        string projectDir = Directory.GetParent(currentDir)?.Parent?.Parent?.FullName;
        // Build full path to the CSV file in the project root
        string dataPath = Path.Combine(projectDir, "turn_data.csv");
        // Load the data
        var data = mlContext.Data.LoadFromTextFile<GameData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true
        );

        // Confirm path
        Console.WriteLine($"CSV data loaded from: {dataPath}");

        /* **********************************************
            * STEP 3: Train/Test Split (80/20)              
            * Random split preserves data distribution     
            * TestFraction: 0.2 → 20% for validation       
         ********************************************** */
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
        var trainSet = split.TrainSet;
        var testSet = split.TestSet;

        // Build the training pipeline
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(GameData.Best_Move_Operation))
     .Append(mlContext.Transforms.Categorical.OneHotEncoding("Luck_Card_Used_Encoded", nameof(GameData.Luck_Card_Used)))
     .Append(mlContext.Transforms.Categorical.OneHotEncoding("In_Lock_Encoded", nameof(GameData.In_Lock)))
     .Append(mlContext.Transforms.Categorical.OneHotEncoding("Opponent_Close_Encoded", nameof(GameData.Opponent_Close)))
     .Append(mlContext.Transforms.Categorical.OneHotEncoding("Move_Direction_Encoded", nameof(GameData.Move_Direction))) 
     .Append(mlContext.Transforms.Concatenate("Features",
         nameof(GameData.Dice_Roll_1),
         nameof(GameData.Dice_Roll_2),
         nameof(GameData.Credits_Before),
         nameof(GameData.Position_Before),
         "Luck_Card_Used_Encoded",
         "In_Lock_Encoded",
         "Opponent_Close_Encoded",
         "Move_Direction_Encoded" 
     ))
     .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features"))
     .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        /* **********************************************
             * STEP 4: Model Training    
             * Fit() method executes the pipeline on data
        ************************************************* */
        Console.WriteLine("===== Training Model =====");
        var model = pipeline.Fit(trainSet);

        /* **********************************************
            * STEP 5: Model Evaluation  
            * Transform test data through model 
            * Calculate quality metrics  
        *********************************************** */
        Console.WriteLine("===== Evaluating Model =====");
        var predictions = model.Transform(testSet);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");
        Console.WriteLine($"LogLoss: {metrics.LogLoss:F4}");

        /* **********************************************
        * STEP 6: Model Persistence                    
        * Save trained model for later use 
        * Includes both model and data schema          
      ********************************************** */
        string modelPath = Path.Combine(projectDir, "RaceToInfinity.zip");
        //Save the model
        mlContext.Model.Save(model, trainSet.Schema, modelPath);
        // Confirm
        Console.WriteLine($"Model saved successfully at: {modelPath}");

        Console.WriteLine("Model training and saving complete! Press any key to EXIT");
        Console.ReadKey();
    }
}
