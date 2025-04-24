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
        /* **********************************************
          * Step 1: Initialise MLContext
          * MLContext is the core object in ML.NET
          * It provides functions for data loading, model training, and evaluation.
           ********************************************** */
        var mlContext = new MLContext();



        /* *******************************************
        * Step 2: Load Data from csv
        * Loads a CSV file (game_data.csv) into an IDataView (ML.NET's data format).
        * Uses the GameData class (not shown in your code but must be defined elsewhere).
        * hasHeader: true → The first row contains column names.
        * separatorChar: ',' → Uses a comma as the delimiter.
        ********************************************* */
        string dataPath = @"C:\Users\RC782\source\repos\game_data.csv";

        var lines = File.ReadLines(dataPath).Take(5);
        foreach (var line in lines)
        {
            Console.WriteLine($"RAW LINE: {line}");
            var columns = line.Split('\t'); // or ',' if comma-separated
            Console.WriteLine($"Columns: {columns.Length}");
        }

            IDataView data = mlContext.Data.LoadFromTextFile<GameData>(
                path: dataPath,
                hasHeader: true,
               separatorChar: ',',         
               allowQuoting: true,          
              allowSparse: false
            );


        /* **********************************************
               * DATA QUALITY CHECK                  
               * Preview first 10 rows for structural integrity
               * Helps identify missing values or format issues
       
        Console.WriteLine("===== First 10 Rows =====");
        var dataPreview = mlContext.Data.CreateEnumerable<GameData>(data, reuseRowObject: false).Take(10);
        foreach (var row in dataPreview)
        {
            Console.WriteLine($"Player_ID: {row.Player_ID}, Dice_Roll_1: {row.Dice_Roll_1}, Dice_Roll_2: {row.Dice_Roll_2}, Math_Operation: {row.Math_Operation}, Move_Direction: {row.Move_Direction}, Move_Value: {row.Move_Value}, Credits_Before: {row.Credits_Before}, Opponent_Interaction: {row.Opponent_Interaction}");
        }

        /* **********************************************
* FULL DATA VISUALIZATION                       *
* Inspect entire dataset (caution with big data)*
********************************************** */
        Console.WriteLine("===== DATA CHECK =====");
        var previewRows = mlContext.Data.CreateEnumerable<GameData>(data, reuseRowObject: false).Take(5);
        foreach (var row in previewRows)
        {
            Console.WriteLine($"Player_ID: {row.Player_ID}, Dice_Roll_1: {row.Dice_Roll_1}, Dice_Roll_2: {row.Dice_Roll_2}");
        }


        /* **********************************************
            * STEP 3: Train/Test Split (80/20)              
            * Random split preserves data distribution     
            * TestFraction: 0.2 → 20% for validation       
         ********************************************** */

        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainSet = split.TrainSet;
            var testSet = split.TestSet;

        /* **********************************************
          * STEP 4: Data Transformation Pipeline         
          * 1. Convert numeric fields to Single type     
          * 2. One-hot encode categorical features       
          * 3. Combine features into single vector      
          * 4. Apply machine learning algorithm          
        ********************************************** */
       
        var pipeline = mlContext.Transforms
       .ReplaceMissingValues("Dice_Roll_1", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
       .Append(mlContext.Transforms.ReplaceMissingValues("Dice_Roll_2", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))

       // One-hot encode string categorical features
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Math_Operation_Encoded", nameof(GameData.Math_Operation)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Move_Direction_Encoded", nameof(GameData.Move_Direction)))
       .Append(mlContext.Transforms.Categorical.OneHotEncoding("Position_Type_Encoded", nameof(GameData.Position_Type)))
       .Append(mlContext.Transforms.Categorical.OneHotEncoding("Luck_Card_Used_Encoded", nameof(GameData.Luck_Card_Used)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Opponent_Interaction_Encoded", nameof(GameData.Opponent_Interaction)))
       .Append(mlContext.Transforms.Categorical.OneHotEncoding("Decision_Taken_Encoded", nameof(GameData.Decision_Taken)))

       // Combine all features into one Features vector
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

       // Add the ML algorithm
                .Append(mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: nameof(GameData.Won_Game),
                    featureColumnName: "Features",
                    numberOfLeaves: 20,
                    numberOfTrees: 100
                ));


        /* **********************************************
             * STEP 5-6: Model Training    
             * Fit() method executes the pipeline on data
             * This is where actual computation happens  *********************************************** */
        Console.WriteLine("===== Training the model =====");
            var model = pipeline.Fit(trainSet);

        /* **********************************************
            * STEP 7: Model Evaluation  
            * Transform test data through model 
            * Calculate quality metrics   ********************************************** */
        try
        {
            Console.WriteLine("===== Evaluating the model =====");
            var predictions = model.Transform(testSet);
            // Convert the prediction Key<UInt32> to boolean (1 -> true, 0 -> false)
            var result = mlContext.Data.CreateEnumerable<GamePrediction>(predictions, reuseRowObject: false)
                                        .Select(pred => new
                                        {
                                            PredictedWin = pred.PredictedWin,
                                            Score = pred.Probability
                                        })
                                        .ToList();
            // Print results to the console
            Console.WriteLine("===== Results =====");
            // Filter predictions with probability > 0.6 and sort descending
            var likelyWins = result
                .Where(r => r.Score > 0.6)
                .OrderByDescending(r => r.Score)
                .ToList();

            Console.WriteLine("===== High-Probability Wins (> 60%) =====");
            foreach (var item in likelyWins)
            {
                Console.WriteLine($"PredictedWin: {item.PredictedWin}, Probability: {item.Score:P2}");
            }

            Console.WriteLine($"Total High-Probability Predictions: {likelyWins.Count}");


            var metrics = mlContext.BinaryClassification.Evaluate(
                    data: predictions,
                    labelColumnName: "Won_Game");

            //Calculates evaluation metrics
            // Overall correctness
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            // Ranking ability
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            // Balance of precision/recall
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");

        }


        /* **********************************************
            * STEP 8: Make Single Prediction  
            * Create prediction engine for real-time usage 
            * Process single instance through model
         *********************************************** */
        // STEP 8: Make Single Prediction (using Transform pipeline)
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
            Decision_Taken = "Moved Forward using Multiply â†’ 9"
        };


        var samples = new List<GameData> { sample };

        // Convert to IDataView
        var sampleDataView = mlContext.Data.LoadFromEnumerable(samples);

        // Run the model pipeline
        var transformed = model.Transform(sampleDataView);

        // Extract prediction result
        var results = mlContext.Data.CreateEnumerable<GamePrediction>(transformed, reuseRowObject: false).ToList();

        // Display result
        Console.WriteLine("===== Single Prediction =====");
        foreach (var result in results)
        {
            Console.WriteLine($"Predicted Win Probability: {result.Probability:P2}");
            Console.WriteLine($"Confidence Threshold: {result.PredictedWin}");
        }


        /* **********************************************
          * STEP 9: Model Persistence                    
          * Save trained model for later use 
          * Includes both model and data schema          
        ********************************************** */
        mlContext.Model.Save(model, trainSet.Schema, "C:\\Users\\RC782\\source\\repos\\MLNetConsoleDemo\\RaceToInfinityModel.zip");
            Console.WriteLine("Press any key to exit.");
            Console.ReadKey();
        }
    }

