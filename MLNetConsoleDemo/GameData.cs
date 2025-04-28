using Microsoft.ML.Data;

namespace MLNetConsoleDemo
{
    public class GameData
    {
        [LoadColumn(0)] public float Dice_Roll_1 { get; set; }
        [LoadColumn(1)] public float Dice_Roll_2 { get; set; }
        [LoadColumn(2)] public float Credits_Before { get; set; }
        [LoadColumn(3)] public float Position_Before { get; set; }
        [LoadColumn(4)] public string Luck_Card_Used { get; set; }  // Yes / No
        [LoadColumn(5)] public string In_Lock { get; set; }         // Yes / No
        [LoadColumn(6)] public string Opponent_Close { get; set; }  // Yes / No
        [LoadColumn(7)] public string Move_Direction { get; set; }
        [LoadColumn(8)] public string Best_Move_Operation { get; set; }  // Label: Add / Subtract / Multiply / Divide
    }
}
