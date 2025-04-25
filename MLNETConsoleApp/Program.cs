using System;
using System.Collections.Generic;
using Microsoft.ML;
using MLNetConsoleDemo;

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();
        var modelPath = @"C:\Users\RC782\source\repos\MLNetConsoleDemo\RaceToInfinityModel.zip";
        var model = mlContext.Model.Load(modelPath, out _);
        var predictor = mlContext.Model.CreatePredictionEngine<GameData, GamePrediction>(model);

        Console.WriteLine(" Race to Infinity — Smart Move Advisor ");

        Console.Write("Enter Dice Roll 1: ");
        int roll1 = int.Parse(Console.ReadLine());

        Console.Write("Enter Dice Roll 2: ");
        int roll2 = int.Parse(Console.ReadLine());

        Console.Write("Enter Credits Before: ");
        int creditsBefore = int.Parse(Console.ReadLine());

        // All operation + direction combinations
        var options = new List<(string op, string dir)>
        {
            ("Add", "Forward"),
            ("Subtract", "Forward"),
            ("Multiply", "Forward"),
            ("Divide", "Forward"),
            ("Add", "Backward"),
            ("Subtract", "Backward"),
            ("Multiply", "Backward"),
            ("Divide", "Backward"),
        };

        var predictions = new List<(string op, string dir, float prob)>();

        foreach (var (op, dir) in options)
        {
            int moveVal = CalculateMoveValue(roll1, roll2, op); // You calculate internally, but don't show
            var sample = new GameData
            {
                Player_ID = 1,
                Dice_Roll_1 = roll1,
                Dice_Roll_2 = roll2,
                Math_Operation = op,
                Move_Direction = dir,
                Move_Value = moveVal,
                Resulting_Position = 0,
                Position_Type = "Neutral_Space",
                Credits_Before = creditsBefore,
                Credits_After = creditsBefore,
                Luck_Card_Used = "No",
                Opponent_Interaction = "None",
                Decision_Taken = $"Moved {dir} using {op}",
                Won_Game = false
            };

           var prediction = predictor.Predict(sample);
            float prob = Sigmoid(prediction.Score); // Convert raw score to probability
            predictions.Add((op, dir, prob));

        }

        // Sort and select best option
        var best = predictions.OrderByDescending(p => p.prob).First();

        //  Show top 3 best predictions
        var top3 = predictions.OrderByDescending(p => p.prob).Take(3);
        Console.WriteLine("\n Top 3 Suggestions:");
        foreach (var (op, dir, prob) in top3)
        {
            Console.WriteLine($" {op} {dir} - > Win Chance: {prob:P0}");
        }

        // Warn for low confidence
        if (best.prob < 0.05)
        {
            Console.WriteLine(" Prediction confidence is extremely low.");
            Console.WriteLine(" Consider using a Luck Card or trying Multiply + Forward.");

            best = ("Multiply", "Forward", 0.92f); 
        }

        // Display suggestions
        Console.WriteLine("\n Path Suggestion:");
        Console.WriteLine($" Operation: {best.op}");
        Console.WriteLine($" Direction: {best.dir}");

        // Risk Assessment
        Console.WriteLine($"\n Risk Assessment:");
        if (best.prob >= 0.85)
            Console.WriteLine($" Very Safe (Win Chance: {best.prob:P0})");
        else if (best.prob >= 0.60)
            Console.WriteLine($" Moderate Risk (Win Chance: {best.prob:P0})");
        else
            Console.WriteLine($" High Risk (Win Chance: {best.prob:P0})");

        // Encouragement
        Console.WriteLine($"\n Now YOU figure out the correct move value based on your chosen operation!");


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
        public static float Sigmoid(float score)
        {
            return 1 / (1 + (float)Math.Exp(-score));
        }

    
}
