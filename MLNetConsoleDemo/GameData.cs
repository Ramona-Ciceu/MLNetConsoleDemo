using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MLNetConsoleDemo
{
    public class GameData
    {
        [LoadColumn(0)] public float Player_ID { get; set; }
        [LoadColumn(1)] public float Dice_Roll_1 { get; set; }
        [LoadColumn(2)] public float Dice_Roll_2 { get; set; }
        [LoadColumn(3)] public string Math_Operation { get; set; }
        [LoadColumn(4)] public string Move_Direction { get; set; }
        [LoadColumn(5)] public float Move_Value { get; set; }
        [LoadColumn(6)] public float Resulting_Position { get; set; }
        [LoadColumn(7)] public string Position_Type { get; set; }
        [LoadColumn(8)] public float Credits_Before { get; set; }
        [LoadColumn(9)] public float Credits_After { get; set; }
        [LoadColumn(10)] public string Luck_Card_Used { get; set; }
        [LoadColumn(11)] public string Opponent_Interaction { get; set; }
        [LoadColumn(12)] public string Decision_Taken { get; set; }
        [LoadColumn(13)] public bool Won_Game { get; set; } // Label
    }


    public class GamePrediction
    {
        [ColumnName("PredictedLabel")] public bool PredictedWin { get; set; }
        [ColumnName("Probability")] public float Probability { get; set; }
    }
}
