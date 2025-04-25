using Xunit;
using Microsoft.ML;
using MLNetConsoleDemo;

public class ModelTests
{
    private readonly string _modelPath = @"C:\Users\RC782\source\repos\MLNetConsoleDemo\RaceToInfinityModel.zip";

    [Fact]
    public void ModelLoadsSuccessfully()
    {
        var mlContext = new MLContext();

        var exception = Record.Exception(() =>
            mlContext.Model.Load(_modelPath, out _));

        Assert.Null(exception);
    }

    [Fact]
    public void ModelPredictsWithinRange()
    {
        var mlContext = new MLContext();
        var model = mlContext.Model.Load(_modelPath, out _);
        var engine = mlContext.Model.CreatePredictionEngine<GameData, GamePrediction>(model);

        var sample = new GameData
        {
            Player_ID = 1,
            Dice_Roll_1 = 8,
            Dice_Roll_2 = 4,
            Math_Operation = "Add",
            Move_Direction = "Forward",
            Move_Value = 12,
            Resulting_Position = 22,
            Position_Type = "Credit_Space",
            Credits_Before = 150,
            Credits_After = 200,
            Luck_Card_Used = "No",
            Opponent_Interaction = "None",
            Decision_Taken = "Moved Forward using Add → 12",
            Won_Game = false
        };

        var prediction = engine.Predict(sample);
        Assert.InRange(prediction.Score, 0, 1);
    }
}
