// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime;
using System.Linq;
using System.Collections.Generic;
using static Microsoft.ML.Runtime.Tools.Console.AttendeeModel;
using Microsoft.ML.Runtime.Api;
using System.Diagnostics;
using System.IO;
using static Microsoft.ML.Runtime.Tools.Console.AmazonModel;

namespace Microsoft.ML.Runtime.Tools.Console
{
    public static class Console
    {
        private static string _metadataFile = @"%USERPROFILE%\Source\Repos\TLC\DataCollector\model-data-file-paths.txt";
        private static string _amznInputRecordFile = @"%USERPROFILE%\OneDrive - Microsoft\models\AmazonReview_100records.csv";
        private static string _atndInputRecordFile = @"%USERPROFILE%\OneDrive - Microsoft\models\AttendeeCountPrediction_100records.csv";

        public static int Main(string[] args)
        { //=> Maml.Main(args);
            int batchSize = 1;
            if (args.Length > 0)
            {
                batchSize = int.Parse(args[0]);
            }

            IHostEnvironment env = new TlcEnvironment();
            var atndBatches = AttendeeModel.Parse(Environment.ExpandEnvironmentVariables(_atndInputRecordFile), batchSize);
            var amznBatches = AmazonModel.Parse(Environment.ExpandEnvironmentVariables(_amznInputRecordFile), batchSize);
            LoadModels(env,
                out IDictionary<string, BatchPredictionEngine<AttendeeData, AttendeeResult>> atndEngines,
                out IDictionary<string, BatchPredictionEngine<AmazonData, AmazonResult>> amznEngines);

            foreach (var engine in atndEngines)
            {
              foreach (var batch in atndBatches)
              {
                var startTime = Stopwatch.GetTimestamp();
                var result = engine.Value.Predict(batch, false);
                var latency = (Stopwatch.GetTimestamp() - startTime) * 1000.0 / Stopwatch.Frequency;
                System.Console.WriteLine("File: {0} Latency: {1} Result: {2}", engine.Key, latency, string.Join(',', result.Select(x => x.Score)));
              }
            }
            foreach (var engine in amznEngines)
            {
              foreach (var batch in amznBatches)
              {
                var startTime = Stopwatch.GetTimestamp();
                var result = engine.Value.Predict(batch, false);
                var latency = (Stopwatch.GetTimestamp() - startTime) * 1000.0 / Stopwatch.Frequency;
                System.Console.WriteLine("File: {0} Latency: {1} Result: {2}", engine.Key, latency, string.Join(',', result.Select(x => x.Score)));
              }
            }

            System.Console.WriteLine("Done");
            return 0;
        }

        private static void LoadModels(IHostEnvironment env,
            out IDictionary<string, BatchPredictionEngine<AttendeeData, AttendeeResult>> atndEngines,
            out IDictionary<string, BatchPredictionEngine<AmazonData, AmazonResult>> amznEngines)
        {
          var numModels = 0;
          var totalStartTime = Stopwatch.GetTimestamp();
          atndEngines = new Dictionary<string, BatchPredictionEngine<AttendeeData, AttendeeResult>>();
          amznEngines = new Dictionary<string, BatchPredictionEngine<AmazonData, AmazonResult>>();
          using (var reader = new StreamReader(Environment.ExpandEnvironmentVariables(_metadataFile)))
          {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
              string[] splits = line.Split();
              var modelPath = Environment.ExpandEnvironmentVariables(splits[0]);

              var startTime = Stopwatch.GetTimestamp();
              if (modelPath.Contains("Attendee"))
              {
                var engine = AttendeeModel.CreateEngine(env, Environment.ExpandEnvironmentVariables(modelPath));
                atndEngines.Add(modelPath, engine);
              }
              else if (modelPath.Contains("Amazon"))
              {
                var engine = AmazonModel.CreateEngine(env, Environment.ExpandEnvironmentVariables(modelPath));
                amznEngines.Add(modelPath, engine);
              }
              var loadingTime = (Stopwatch.GetTimestamp() - startTime) * 1000.0 / Stopwatch.Frequency;
              System.Console.WriteLine("Load {0} in {1} ms", modelPath, loadingTime);
              // TODO: Other type of models

              numModels++;
            }
          }

          var totalLoadingTime = (Stopwatch.GetTimestamp() - totalStartTime) * 1000.0 / Stopwatch.Frequency;
          System.Console.WriteLine("Load {0} models in {1} ms", numModels, totalLoadingTime);
        }
    }
}
