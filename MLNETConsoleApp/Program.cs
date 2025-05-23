﻿using System;
using Microsoft.ML;
using MLNetConsoleDemo;

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Load the trained Move Prediction model
        //Get current directory where the app is running 
        // Get current working directory
        string currentDir = Directory.GetCurrentDirectory();
        // Navigate up to the project root (bin → Debug → net7.0 → project folder)
        string projectDir = Directory.GetParent(currentDir)?.Parent?.Parent?.FullName;
        // Now go to MLNetConsoleDemo (this is the correct folder name!)
        string modelSourceDir = Path.Combine(projectDir, "MLNetConsoleDemo");
        // Build full path to the model file in the project root
        string modelPath = Path.Combine(projectDir, "RaceToInfinity.zip");
        // Load the model
        var model = mlContext.Model.Load(modelPath, out _);
        // Create prediction engine
        var predictor = mlContext.Model.CreatePredictionEngine<GameData, GamePrediction>(model);
        // Optional: confirm the model path
        Console.WriteLine($"Model loaded from: {modelPath}");


        Console.WriteLine("Race to Infinity — Smart Move Advisor Simulation\n");

        bool keepPlaying = true;

        while (keepPlaying)
        {
            Console.WriteLine("Enter current turn information:");

            // Get user inputs
            int dice1 = GetIntInput("Dice Roll 1 (1-12): ", 1, 12);
            int dice2 = GetIntInput("Dice Roll 2 (1-12): ", 1, 12);
            int credits = GetIntInput("Credits Before Move (0-250): ", 0, 250);
            int position = GetIntInput("Board Position (0-70): ", 0, 70);
            string luckUsed = GetYesNoInput("Luck Card Used? (yes/no): ");
            string inLock = GetYesNoInput("In Lock? (yes/no): ");
            string opponentNearby = GetYesNoInput("Opponent Nearby? (yes/no): ");
            string moveDirection = GetDirectionInput("Expected Move Direction? (forward/backward): ");


            var turnData = new GameData
            {
                Dice_Roll_1 = dice1,
                Dice_Roll_2 = dice2,
                Credits_Before = credits,
                Position_Before = position,
                Luck_Card_Used = luckUsed,
                In_Lock = inLock,
                Opponent_Close = opponentNearby,
                Move_Direction = moveDirection,
                Best_Move_Operation = "" // Placeholder, not needed during prediction
            };

            var prediction = predictor.Predict(turnData);

            // Get class labels in the same order used during training
            string[] moveLabels = { "Add", "Subtract", "Multiply", "Divide" };
            // Estimate move value (you can later base this on predicted operation)
            int moveValue = dice1 + dice2;
            int forwardPosition = (int)turnData.Position_Before + moveValue;
            int backwardPosition = (int)turnData.Position_Before - moveValue;


            // Ensure positions stay within board boundaries
            forwardPosition = Math.Min(forwardPosition, 70);
            backwardPosition = Math.Max(backwardPosition, 0);

            // Basic logic: prefer whichever is closer to center (e.g. 60 or 70)
            int distanceForward = Math.Abs(60 - forwardPosition);
            int distanceBackward = Math.Abs(60 - backwardPosition);

            string bestPath = distanceForward <= distanceBackward ? "Forward" : "Backward";


            // Find the predicted index and confidence
            int predictedIndex = Array.IndexOf(moveLabels, prediction.PredictedMove);
            float confidence = prediction.Score[predictedIndex];

            // Determine risk level based on confidence
            string riskLevel;
            if (confidence >= 0.8f) riskLevel = "Low Risk";
            else if (confidence >= 0.5f) riskLevel = "Medium Risk";
            else riskLevel = "High Risk";

            // Display prediction and evaluation
            Console.WriteLine($"\n=== AI Suggestion ===");
            Console.WriteLine($"Predicted Best Move: {prediction.PredictedMove}");
            Console.WriteLine($"Model Confidence: {confidence:P1}");
            Console.WriteLine($"Risk Indicator: {riskLevel}");
            Console.WriteLine($"Expected Move Direction: {moveDirection}");
            Console.WriteLine($"Suggested Path Based on Position: {bestPath}");
            Console.WriteLine("======================\n");


            // Ask if they want to simulate another turn
            Console.Write("Do you want to simulate another turn? (yes/no): ");
            string again = Console.ReadLine().Trim().ToLower();
            keepPlaying = (again == "yes");
        }

        Console.WriteLine("\nSimulation complete. Press any key to exit.");
        Console.ReadKey();
    }

    // Helper method: Get integer input within a range
    static int GetIntInput(string prompt, int min, int max)
    {
        int value;
        bool valid;
        do
        {
            Console.Write(prompt);
            valid = int.TryParse(Console.ReadLine(), out value) && value >= min && value <= max;
            if (!valid)
            {
                Console.WriteLine($"Please enter a valid number between {min} and {max}.");
            }
        } while (!valid);
        return value;
    }

    // Helper method: Get yes/no input
    static string GetYesNoInput(string prompt)
    {
        string input;
        do
        {
            Console.Write(prompt);
            input = Console.ReadLine().Trim().ToLower();
            if (input != "yes" && input != "no")
            {
                Console.WriteLine("Please enter 'yes' or 'no'.");
            }
        } while (input != "yes" && input != "no");

        return char.ToUpper(input[0]) + input.Substring(1); // Make "Yes"/"No"
    }

    // Helper method: Get Move Direction input
    static string GetDirectionInput(string prompt)
    {
        string input;
        do
        {
            Console.Write(prompt);
            input = Console.ReadLine().Trim().ToLower();
            if (input != "forward" && input != "backward")
            {
                Console.WriteLine("Please enter 'forward' or 'backward'.");
            }
        } while (input != "forward" && input != "backward");

        return char.ToUpper(input[0]) + input.Substring(1); // Make "Forward"/"Backward"
    }
}
