using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Microsoft.ML.Runtime.Tools.Console
{
    public sealed class AmazonModel
    {
        public static BatchPredictionEngine<AmazonData, AmazonResult> CreateEngine(IHostEnvironment env, string modelPath)
        {
            using (var stream = File.OpenRead(modelPath))
            {
                return env.CreateBatchPredictionEngine<AmazonData, AmazonResult>(stream, false, InputSchema(), OutputSchema());
            }
        }

        private static SchemaDefinition InputSchema()
        {
            var amazonInputSchema = SchemaDefinition.Create(typeof(AmazonData));
            amazonInputSchema["ReviewerID"].ColumnType = TextType.Instance;
            amazonInputSchema["Asin"].ColumnType = TextType.Instance;
            amazonInputSchema["Asin"].ColumnName = "asin";
            amazonInputSchema["ReviewerName"].ColumnType = TextType.Instance;
            amazonInputSchema["Helpful"].ColumnType = TextType.Instance;
            amazonInputSchema["Text"].ColumnType = TextType.Instance;
            amazonInputSchema["Label"].ColumnType = NumberType.R4;
            amazonInputSchema["Summary"].ColumnType = TextType.Instance;
            amazonInputSchema["UnixReviewTime"].ColumnType = TextType.Instance;
            amazonInputSchema["ReviewTime"].ColumnType = NumberType.R4;
            return amazonInputSchema;
        }

        public sealed class AmazonData
        {
            public string ReviewerID;
            public string Asin;
            public string ReviewerName;
            public string Helpful;
            public string Text;
            public float Label;
            public string Summary;
            public string UnixReviewTime;
            public float ReviewTime;
        }

        private static SchemaDefinition OutputSchema()
        {
            return SchemaDefinition.Create(typeof(AmazonResult));
        }

        public sealed class AmazonResult
        {
            public float Score;
            public float Probability;
            public override string ToString()
            {
                return Score.ToString();
            }
        }

        public static List<AmazonData[]> Parse(string filePath, int batchSize = 1)
        {
            var batchList = new List<AmazonData[]>();
            var numTotalRecords = 0;
            using (StreamReader reader = new StreamReader(Environment.ExpandEnvironmentVariables(filePath)))
            {
                string line;
                int idx = 0;

                var batch = new AmazonData[batchSize];
                while ((line = reader.ReadLine()) != null)
                {
                    batch[idx] = ParseLine(line);
                    numTotalRecords++;
                    idx++;

                    if (idx == batchSize)
                    {
                        batchList.Add(batch);
                        //Console.WriteLine("{0}st batch (size: {1}) of {2} has been added", batchList.Count, batchSize, typeof(AmazonModel).Name);
                        batch = new AmazonData[batchSize];
                        idx = 0;
                    }
                }

                if (idx > 0 && idx < batchSize)
                {
                    //Console.WriteLine("Remaining {0} records of {1} as the last batch", idx, typeof(AmazonModel).Name);
                    batchList.Add(batch); // Add remaining batch
                }
            }

            System.Console.WriteLine("{0} records have been loaded into {1} batches", numTotalRecords, batchList.Count);
            return batchList;
        }

        public static AmazonData ParseLine(string line, bool readAll = true)
        {
            string[] tokens = new string[9];

            int fieldToAnalyzeStart = 0;
            int fId = 0;

            int currentField = 0;
            bool skipFieldSep = false;
            int i = 0;
            int countSeqLen = 0;
            int len = line.Length;

            int stackTot = 0;
            int stackTop = 0;
            int stackBase = 0;

            // parse initial '"'
            while (line[countSeqLen] == '"')
            {
                countSeqLen++;
            }
            i = countSeqLen;

            stackTot = stackTop = countSeqLen;
            stackBase = countSeqLen > 0 ? 1 : 0;

            while (i < len)
            {
                countSeqLen = 0;
                while (i + countSeqLen < len && line[i + countSeqLen] == '"') countSeqLen++;
                i += countSeqLen;

                int prevQuoteSeqLen = countSeqLen;
                if (countSeqLen > 0)
                {
                    if (countSeqLen > stackTot)
                    {
                        stackTot |= countSeqLen;
                        stackTop = countSeqLen & (~countSeqLen >> 1);
                    }
                    else if (countSeqLen == stackTop / 2)
                    {
                        stackTot -= stackTop;
                        stackTot -= countSeqLen;
                        stackTop = countSeqLen / 2;
                    }
                    else
                    {
                        stackTot ^= countSeqLen;
                        stackTop = stackTot & (~stackTot >> 1);
                    }
                    if (!Functions.IsContigBitmask(stackTot))
                    {
                        break;
                    }
                }
                skipFieldSep = stackTop > stackBase;

                if (!skipFieldSep && line[i] == ',')
                {
                    if (readAll || fId == 4)
                    {
                        tokens[fId] = line.Substring(fieldToAnalyzeStart + prevQuoteSeqLen + 1,
                            i - fieldToAnalyzeStart - prevQuoteSeqLen * 2 - 1);
                    }
                    fId++;

                    currentField++;
                    fieldToAnalyzeStart = i;
                }
                i++;
            }

            if (tokens[8] == null)
            {
                tokens[fId] = line.Substring(fieldToAnalyzeStart + 1, line.Length - fieldToAnalyzeStart - 1);
            }

            if (readAll)
            {
                var amazonData = new AmazonData
                {
                    ReviewerID = tokens[0],
                    Asin = tokens[1],
                    ReviewerName = tokens[2],
                    Helpful = tokens[3],
                    Text = tokens[4],
                    Label = int.Parse(tokens[5]),
                    Summary = tokens[6],
                    UnixReviewTime = tokens[7],
                    ReviewTime = int.Parse(tokens[8]),
                };
                return amazonData;
            }
            else
            {
                var amazonData = new AmazonData
                {
                    ReviewerID = "",
                    Asin = "",
                    ReviewerName = "",
                    Helpful = "",
                    Text = tokens[4],
                    Label = -1,
                    Summary = "",
                    UnixReviewTime = "",
                    ReviewTime = -1
                };
                return amazonData;
            }

        }
    }
}
