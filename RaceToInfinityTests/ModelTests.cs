using Xunit;
using Microsoft.ML;
using MLNetConsoleDemo;

public class ModelTests
{
    private readonly string _modelPath = @"C:\Users\RC782\source\repos\RaceToInfinity_MovePredictor.zip"; // <-- Correct model

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
            Best_Move_Operation = "" // Placeholder, not needed at prediction
        };

        var prediction = engine.Predict(sample);

        // Valid moves
        var validMoves = new[] { "Add", "Subtract", "Multiply", "Divide" };

        Assert.Contains(prediction.PredictedMove, validMoves);
    }
}
