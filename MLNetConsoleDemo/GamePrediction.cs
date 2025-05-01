using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML.Data;

namespace MLNetConsoleDemo
{
    public class GamePrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedMove { get; set; }
        [ColumnName("Score")]
        public float[] Score { get; set; }  // probability scores for each class
    }
}
