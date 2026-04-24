using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;


    // 1. Define the Data Schema
    public class RawData
    {
        public float Feature0 { get; set; } // Continuous
        public float Feature1 { get; set; } // Continuous
        public float Feature2 { get; set; } // Categorical (0-4)
        public bool Label { get; set; }
    }

public class Prediction
{
    public bool PredictedLabel { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}

public class Train
{
    public static void Yyy()
    {
        var mlContext = new MLContext(seed: 42);
        var rng = new Random(42);

        // 1. Generate Synthetic Data
        var list = new List<RawData>();
        for (int i = 0; i < 500; i++)
        {
            list.Add(new RawData
            {
                Feature0 = (float)rng.NextDouble(),
                Feature1 = (float)rng.NextDouble(),
                Feature2 = (float)rng.Next(0, 5),
                Label = rng.Next(0, 2) == 1
            });
        }
        var dataView = mlContext.Data.LoadFromEnumerable(list);

        //// 2. The Correct Pipeline
        //var pipeline = mlContext.Transforms.Conversion
        //    // Step A: Map to Key (this creates the categorical metadata)
        //    .MapValueToKey(nameof(RawData.Feature2))
        //    // Step B: Convert Key to Float (this makes it compatible with Concatenate)
        //    // ML.NET preserves the "Key" metadata during this specific conversion!
        //    .Append(mlContext.Transforms.Conversion.ConvertType(nameof(RawData.Feature2), outputKind: DataKind.Single))
        //    // Step C: Concatenate all floats into one vector
        //    .Append(mlContext.Transforms.Concatenate("Features",
        //        nameof(RawData.Feature0),
        //        nameof(RawData.Feature1),
        //        nameof(RawData.Feature2)))
        //    // Step D: Train LightGBM
        //    .Append(mlContext.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
        //    {
        //        NumberOfIterations = 10,
        //        NumberOfLeaves = 8,
        //        UseCategoricalSplit = true,
        //        FeatureColumnName = "Features",
        //        LabelColumnName = nameof(RawData.Label)
        //    }));

        var pipeline =
    mlContext.Transforms.Conversion.MapValueToKey("Feature2Key", nameof(RawData.Feature2))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("Feature2Float", "Feature2Key"))
    .Append(mlContext.Transforms.Concatenate("Features",
        nameof(RawData.Feature0),
        nameof(RawData.Feature1),
        "Feature2Float"))
    .Append(mlContext.BinaryClassification.Trainers.LightGbm(new LightGbmBinaryTrainer.Options
    {
        FeatureColumnName = "Features",
        LabelColumnName = nameof(RawData.Label),
        UseCategoricalSplit = true,
        NumberOfIterations = 10,
        NumberOfLeaves = 8
    }));

        // 3. Train
        var model = pipeline.Fit(dataView);

        // 4. Save to lgbm.txt
        // We use this logic to drill past the 'Calibrator' wrapper ML.NET adds
        var transformer = model.LastTransformer as ISingleFeaturePredictionTransformer<object>;
        var lgbmParams = transformer.Model as LightGbmBinaryModelParameters;

    }
}
