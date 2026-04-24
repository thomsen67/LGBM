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
        // optional: force console logging
        mlContext.Log += (o, e) =>
        {
            Console.WriteLine(e.Message);
        };

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
        NumberOfLeaves = 8,
        //Verbose = true, not informative
    }));

        // 3. Train
        var model = pipeline.Fit(dataView);

        mlContext.Model.Save(model, dataView.Schema, @"c:\Thomas\Desktop\gekko\testing\DREAM\LGBM_TTH_Repo\lgbm.zip");

        // 4. Save to lgbm.txt
        // We use this logic to drill past the 'Calibrator' wrapper ML.NET adds
        var transformer = model.LastTransformer as ISingleFeaturePredictionTransformer<object>;
        var lgbmParams = transformer.Model as LightGbmBinaryModelParameters;

        
        // model = output of pipeline.Fit(...)
        var predictionEngine = mlContext.Model.CreatePredictionEngine<RawData, Prediction>(model);

        var sample = new RawData
        {
            Feature0 = 0.5f,
            Feature1 = 0.2f,
            Feature2 = 3 // categorical value (same domain as training!)
        };

        var result = predictionEngine.Predict(sample);

        if (true)
        {
            //Faster: running in parallel
            var testData = mlContext.Data.LoadFromEnumerable(new[]
{
    new RawData { Feature0 = 0.1f, Feature1 = 0.2f, Feature2 = 1 },
    new RawData { Feature0 = 0.4f, Feature1 = 0.6f, Feature2 = 3 }
});

            var predictions = model.Transform(testData);

            var results = mlContext.Data.CreateEnumerable<Prediction>(predictions, reuseRowObject: false);

            foreach (var p in results)
            {
                Console.WriteLine(p.PredictedLabel);
            }
        }

        // ── 5. Evaluation statistics ─────────────────────────────────────────────────

        // Use the full training data for in-sample metrics (or swap for a held-out test set)
        var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 42);
        var trainSet = split.TrainSet;
        var testSet = split.TestSet;

        // Re-fit on train split only, evaluate on test split
        var modelForEval = pipeline.Fit(trainSet);
        var evaluated = modelForEval.Transform(testSet);

        var metrics = mlContext.BinaryClassification.Evaluate(
            evaluated,
            labelColumnName: nameof(RawData.Label));

        Console.WriteLine("\n── Binary Classification Metrics ──────────────────");
        Console.WriteLine($"  Accuracy                : {metrics.Accuracy:F4}");
        Console.WriteLine($"  AUC (ROC)               : {metrics.AreaUnderRocCurve:F4}");
        Console.WriteLine($"  AUC (Precision-Recall)  : {metrics.AreaUnderPrecisionRecallCurve:F4}");
        Console.WriteLine($"  F1 Score                : {metrics.F1Score:F4}");
        Console.WriteLine($"  Precision               : {metrics.PositivePrecision:F4}");
        Console.WriteLine($"  Recall                  : {metrics.PositiveRecall:F4}");
        Console.WriteLine($"  Negative Precision      : {metrics.NegativePrecision:F4}");
        Console.WriteLine($"  Negative Recall         : {metrics.NegativeRecall:F4}");
        Console.WriteLine($"  Log Loss                : {metrics.LogLoss:F4}");
        Console.WriteLine($"  Log Loss Reduction      : {metrics.LogLossReduction:F4}");

        Console.WriteLine("\n── Confusion Matrix ────────────────────────────────");
        var cm = metrics.ConfusionMatrix;
        Console.WriteLine($"  True  Positives  : {cm.GetFormattedConfusionTable()}");
        Console.WriteLine(cm.GetFormattedConfusionTable());

        // ── 6. Feature importance ─────────────────────────────────────────────────────
        // Drill through the calibrator wrapper to get the LightGBM model
        var lastTransformer = modelForEval.LastTransformer
            as ISingleFeaturePredictionTransformer<object>;
        var lgbmModel = lastTransformer?.Model as LightGbmBinaryModelParameters;

        //if (lgbmModel != null)
        //{
        //    var gains = lgbmModel.GetFeatureWeights(normalizeWeights: true);

        //    // Feature names in the same order as Concatenate("Features", ...)
        //    var featureNames = new[] { "Feature0", "Feature1", "Feature2Float" };

        //    Console.WriteLine("\n── Feature Importance (normalized gain) ────────────");
        //    for (int i = 0; i < gains.Length && i < featureNames.Length; i++)
        //        Console.WriteLine($"  {featureNames[i],-20}: {gains[i]:F4}");
        //}


    }
}
