using Microsoft.ML;
using Microsoft.ML.Data;
using MLNetConsoleDemo;
using System;
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
        string dataPath = @"C:\Users\RC782\source\repos\turn_data.csv"; 
        var data = mlContext.Data.LoadFromTextFile<GameData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ',',
            allowQuoting: true
        );

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
        mlContext.Model.Save(model, trainSet.Schema, @"C:\Users\RC782\source\repos\RaceToInfinity.zip");

        Console.WriteLine("Model training and saving complete! Press any key to EXIT");
        Console.ReadKey();
    }
}
