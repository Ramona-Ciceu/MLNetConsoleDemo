using Xunit;
using Microsoft.ML;
using System.IO;
using System.Linq;
using MLNetConsoleDemo;

public class DataTests
{
    private readonly string _dataPath = @"C:\Users\RC782\source\repos\game_data.csv";
    [Fact]
    public void TestCsvHasValidColumns()
    {
        // Arrange
        var expectedColumns = new[] {
        "Player_ID", "Dice_Roll_1", "Dice_Roll_2",
        "Math_Operation", "Move_Direction", "Move_Value",
        "Resulting_Position", "Position_Type", "Credits_Before",
        "Credits_After", "Luck_Card_Used", "Opponent_Interaction",
        "Decision_Taken", "Won_Game"
    };

        // Act
        var firstLine = File.ReadLines(_dataPath).First();
        var actualColumns = firstLine
            .Split(',')
            .Select(col => col.Trim('"')) // ✅ fix for quote-wrapped headers
            .ToArray();

        // Assert
        Assert.Equal(expectedColumns, actualColumns);
    }



    [Fact]
    public void TestWonGameColumnHasBooleanValues()
    {
        // Arrange
        var mlContext = new MLContext();
        var data = mlContext.Data.LoadFromTextFile<GameData>(_dataPath, hasHeader: true, separatorChar: ',', allowQuoting: true);

        // Act
        var winValues = mlContext.Data.CreateEnumerable<GameData>(data, reuseRowObject: false)
            .Select(d => d.Won_Game)
            .Distinct()
            .ToList();

        // Assert
        Assert.All(winValues, v => Assert.True(v == true || v == false));
    }
}
