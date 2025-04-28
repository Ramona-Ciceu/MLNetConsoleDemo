using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.ML.Data;

namespace MLNetConsoleDemo
{
    // Class to hold the prediction result
    public class GamePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedWin { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }

        // This will be calculated from the raw score using Sigmoid
        public float Probability => Program.Sigmoid(Score); // Converts raw score to probability
    }
}

