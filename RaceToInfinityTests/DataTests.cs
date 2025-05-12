using Xunit;
using Microsoft.ML;
using System.IO;
using System.Linq;
using MLNetConsoleDemo;

public class DataTests
{
    private readonly string _dataPath;

    public DataTests()
    {
        // Get bin/debug/net7.0 directory
        string currentDir = Directory.GetCurrentDirectory();

        // Go up 3 levels to project root
        string projectDir = Directory.GetParent(currentDir)?.Parent?.Parent?.FullName;

        // Set full path to CSV in the project root
        _dataPath = Path.Combine(projectDir, "turn_data.csv");
    }

    [Fact]
    public void TestCsvHasValidColumns()
    {
        // Arrange
        var expectedColumns = new[] {
            "Dice_Roll_1", "Dice_Roll_2", "Credits_Before",
            "Position_Before", "Luck_Card_Used", "In_Lock",
            "Opponent_Close", "Move_Direction", "Best_Move_Operation"
        };

        // Act
        var firstLine = File.ReadLines(_dataPath).First();
        var actualColumns = firstLine
            .Split(',')
            .Select(col => col.Trim('"'))
            .ToArray();

        // Assert
        Assert.Equal(expectedColumns, actualColumns);
    }

    [Fact]
    public void TestBestMoveOperationHasValidValues()
    {
        // Arrange
        var mlContext = new MLContext();
        var data = mlContext.Data.LoadFromTextFile<GameData>(_dataPath, hasHeader: true, separatorChar: ',', allowQuoting: true);

        // Act
        var moveValues = mlContext.Data.CreateEnumerable<GameData>(data, reuseRowObject: false)
            .Select(d => d.Best_Move_Operation)
            .Distinct()
            .ToList();

        var validMoves = new[] { "Add", "Subtract", "Multiply", "Divide" };

        // Assert
        Assert.All(moveValues, move => Assert.Contains(move, validMoves));
    }
}
