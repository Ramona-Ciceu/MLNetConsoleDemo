using Xunit;
using Microsoft.ML;
using MLNetConsoleDemo;
using System.IO;

public class ModelTests
{
    private readonly string _modelPath;

    public ModelTests()
    {
        // Get the bin/debug/net7.0 directory
        string currentDir = Directory.GetCurrentDirectory();

        // Navigate up to the project root (3 levels)
        string projectDir = Directory.GetParent(currentDir)?.Parent?.Parent?.FullName;

        // Build full path to the model file in the project root
        _modelPath = Path.Combine(projectDir, "RaceToInfinity.zip");
    }

    [Fact]
    public void ModelLoadsSuccessfully()
    {
        var mlContext = new MLContext();

        var exception = Record.Exception(() =>
            mlContext.Model.Load(_modelPath, out _));

        Assert.Null(exception);
    }

    [Fact]
    public void ModelPredictsValidMove()
    {
        var mlContext = new MLContext();
        var model = mlContext.Model.Load(_modelPath, out _);
        var engine = mlContext.Model.CreatePredictionEngine<GameData, GamePrediction>(model);

        var sample = new GameData
        {
            Dice_Roll_1 = 6,
            Dice_Roll_2 = 5,
            Credits_Before = 140,
            Position_Before = 17,
            Luck_Card_Used = "No",
            In_Lock = "No",
            Opponent_Close = "Yes",
            Best_Move_Operation = "" // Not needed for prediction
        };

        var prediction = engine.Predict(sample);

        var validMoves = new[] { "Add", "Subtract", "Multiply", "Divide" };

        Assert.Contains(prediction.PredictedMove, validMoves);
    }
}
