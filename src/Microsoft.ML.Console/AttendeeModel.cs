using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Text;
using System.Linq;
using System.IO;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Tools.Console
{
    public sealed class AttendeeModel
    {
        public static BatchPredictionEngine<AttendeeData, AttendeeResult> CreateEngine(IHostEnvironment env, string modelPath)
        {
            using (var stream = File.OpenRead(modelPath))
            {
                return env.CreateBatchPredictionEngine<AttendeeData, AttendeeResult>(stream, false, InputSchema(), OutputSchema());
            }
        }

        private static SchemaDefinition InputSchema()
        {
            var atndSchema = SchemaDefinition.Create(typeof(AttendeeData));
            atndSchema["ProgramOwner"].ColumnType = TextType.Instance;
            atndSchema["Timezone"].ColumnType = TextType.Instance;
            atndSchema["Timezonemapping"].ColumnType = TextType.Instance;
            atndSchema["Timezonemapping"].ColumnName = "timezonemapping";
            atndSchema["Area"].ColumnType = TextType.Instance;
            atndSchema["Country"].ColumnType = TextType.Instance;
            atndSchema["LocalizationLanguage"].ColumnType = TextType.Instance;
            atndSchema["StartDateDayOfWeek"].ColumnType = TextType.Instance;
            atndSchema["StartDateHrofDay"].ColumnType = NumberType.R4;
            atndSchema["StartDateMonthOfYear"].ColumnType = NumberType.R4;
            atndSchema["Duration"].ColumnType = NumberType.R4;
            atndSchema["DaysBetweenCreationandWebinar"].ColumnType = NumberType.R4;
            atndSchema["RegisteredCount"].ColumnType = NumberType.R4;
            atndSchema["Label"].ColumnType = NumberType.R4;
            atndSchema["SearchPublishToSearch"].ColumnType = BoolType.Instance;
            atndSchema["SearchPublishToSearch"].ColumnName = "Search_PublishToSearch";
            atndSchema["CeOnPremise"].ColumnType = NumberType.R4;
            atndSchema["CeOnPremise"].ColumnName = "c+e_on_premise";

            atndSchema["CloudPlatform"].ColumnType = NumberType.R4;
            atndSchema["CloudPlatform"].ColumnName = "cloud_platform";
            atndSchema["DeveloperTools"].ColumnType = NumberType.R4;
            atndSchema["DeveloperTools"].ColumnName = "developer_tools";
            atndSchema["DevicesXbox"].ColumnType = NumberType.R4;
            atndSchema["DevicesXbox"].ColumnName = "devices_&_xbox";

            atndSchema["DynamicsCloud"].ColumnType = NumberType.R4;
            atndSchema["DynamicsCloud"].ColumnName = "dynamics_cloud";
            atndSchema["DynamicsOnPremise"].ColumnType = NumberType.R4;
            atndSchema["DynamicsOnPremise"].ColumnName = "dynamics_on_premise";
            atndSchema["EnterpriseSolutions"].ColumnType = NumberType.R4;
            atndSchema["EnterpriseSolutions"].ColumnName = "enterprise_solutions";
            atndSchema["Hololens"].ColumnType = NumberType.R4;
            atndSchema["Hololens"].ColumnName = "hololens";
            atndSchema["MicrosoftAzure"].ColumnType = NumberType.R4;
            atndSchema["MicrosoftAzure"].ColumnName = "microsoft_azure";
            atndSchema["MicrosoftDynamics"].ColumnType = NumberType.R4;
            atndSchema["MicrosoftDynamics"].ColumnName = "microsoft_dynamics";
            atndSchema["Office"].ColumnType = NumberType.R4;
            atndSchema["Office"].ColumnName = "office";
            atndSchema["OtherProductsNotListedAbove"].ColumnType = NumberType.R4;
            atndSchema["OtherProductsNotListedAbove"].ColumnName = "other_products_(not_listed_above)";
            atndSchema["ProdUncategorized"].ColumnType = NumberType.R4;
            atndSchema["ProdUncategorized"].ColumnName = "Prod_uncategorized";
            atndSchema["ServerManagement"].ColumnType = NumberType.R4;
            atndSchema["ServerManagement"].ColumnName = "server_&_management";

            atndSchema["Skype"].ColumnType = NumberType.R4;
            atndSchema["Skype"].ColumnName = "skype";
            atndSchema["SmallBusinessSolutions"].ColumnType = NumberType.R4;
            atndSchema["SmallBusinessSolutions"].ColumnName = "small_business_solutions";
            atndSchema["Surface"].ColumnType = NumberType.R4;
            atndSchema["Surface"].ColumnName = "surface";
            atndSchema["Windows"].ColumnType = NumberType.R4;
            atndSchema["Windows"].ColumnName = "windows";
            atndSchema["WindowsPcTablet"].ColumnType = NumberType.R4;
            atndSchema["WindowsPcTablet"].ColumnName = "windows_(pc/tablet)";
            atndSchema["WindowsEmbeddedIot"].ColumnType = NumberType.R4;
            atndSchema["WindowsEmbeddedIot"].ColumnName = "windows_embedded_(iot)";
            atndSchema["WindowsPhone"].ColumnType = NumberType.R4;
            atndSchema["WindowsPhone"].ColumnName = "windows_phone";
            atndSchema["WindowsVolumeLicensing"].ColumnType = NumberType.R4;
            atndSchema["WindowsVolumeLicensing"].ColumnName = "windows_volume_licensing";
            atndSchema["BusinessDecisionMaker"].ColumnType = NumberType.R4;
            atndSchema["BusinessDecisionMaker"].ColumnName = "business_decision_maker";
            atndSchema["BusinessProfessionals"].ColumnType = NumberType.R4;
            atndSchema["BusinessProfessionals"].ColumnName = "business_professionals";
            atndSchema["Consumers"].ColumnType = NumberType.R4;
            atndSchema["Consumers"].ColumnName = "consumers";
            atndSchema["Developers"].ColumnType = NumberType.R4;
            atndSchema["Developers"].ColumnName = "developers";
            atndSchema["Educators"].ColumnType = NumberType.R4;
            atndSchema["Educators"].ColumnName = "educators";
            atndSchema["ForHome"].ColumnType = NumberType.R4;
            atndSchema["ForHome"].ColumnName = "for_home";
            atndSchema["Government"].ColumnType = NumberType.R4;
            atndSchema["Government"].ColumnName = "government";
            atndSchema["InformationWorker"].ColumnType = NumberType.R4;
            atndSchema["InformationWorker"].ColumnName = "information_worker";
            atndSchema["ItDecisionMaker"].ColumnType = NumberType.R4;
            atndSchema["ItDecisionMaker"].ColumnName = "it_decision_maker";
            atndSchema["ItProfessionals"].ColumnType = NumberType.R4;
            atndSchema["ItProfessionals"].ColumnName = "it_professionals";
            atndSchema["OtherAudience"].ColumnType = NumberType.R4;
            atndSchema["OtherAudience"].ColumnName = "other_audience";
            atndSchema["Partners"].ColumnType = NumberType.R4;
            atndSchema["Partners"].ColumnName = "partners";
            atndSchema["Students"].ColumnType = NumberType.R4;
            atndSchema["Students"].ColumnName = "students";
            atndSchema["TAUncategorized"].ColumnType = NumberType.R4;
            atndSchema["TAUncategorized"].ColumnName = "TA_uncategorized";
            return atndSchema;
        }

        private static SchemaDefinition OutputSchema()
        {
            return SchemaDefinition.Create(typeof(AttendeeResult));
        }

        public sealed class AttendeeData
        {
            public string ProgramOwner;
            public string Timezone;
            public string Timezonemapping;
            public string Area;
            public string Country;
            public string LocalizationLanguage;
            public string StartDateDayOfWeek;
            public float StartDateHrofDay;
            public float StartDateMonthOfYear;
            public float Duration;
            public float DaysBetweenCreationandWebinar;
            public float RegisteredCount;
            public float Label;
            public bool SearchPublishToSearch;
            public float CeOnPremise;
            public float CloudPlatform;
            public float DeveloperTools;
            public float DevicesXbox;
            public float DynamicsCloud;
            public float DynamicsOnPremise;
            public float EnterpriseSolutions;
            public float Hololens;
            public float MicrosoftAzure;
            public float MicrosoftDynamics;
            public float Office;
            public float OtherProductsNotListedAbove;
            public float ProdUncategorized;
            public float ServerManagement;
            public float Skype;
            public float SmallBusinessSolutions;
            public float Surface;
            public float Windows;
            public float WindowsPcTablet;
            public float WindowsEmbeddedIot;
            public float WindowsPhone;
            public float WindowsVolumeLicensing;
            public float BusinessDecisionMaker;
            public float BusinessProfessionals;
            public float Consumers;
            public float Developers;
            public float Educators;
            public float ForHome;
            public float Government;
            public float InformationWorker;
            public float ItDecisionMaker;
            public float ItProfessionals;
            public float OtherAudience;
            public float Partners;
            public float Students;
            public float TAUncategorized;
        }

        public sealed class AttendeeResult
        {
            public float Score;
            public override string ToString()
            {
                return Score.ToString();
            }
        }

        public static List<AttendeeData[]> Parse(string filePath, int batchSize = 1)
        {
            var batchList = new List<AttendeeData[]>();
            var numTotalRecords = 0;
            using (StreamReader reader = new StreamReader(Environment.ExpandEnvironmentVariables(filePath)))
            {
                string line;
                int idx = 0;

                var batch = new AttendeeData[batchSize];
                while ((line = reader.ReadLine()) != null)
                {
                    batch[idx] = ParseLine(line);
                    numTotalRecords++;
                    idx++;

                    if (idx == batchSize)
                    {
                        batchList.Add(batch);
                        //Console.WriteLine("{0}st batch (size: {1}) of {2} has been added", batchList.Count, batchSize, typeof(AttendeeModel).Name);
                        batch = new AttendeeData[batchSize];
                        idx = 0;
                    }
                }

                if (idx > 0 && idx < batchSize)
                {
                    //Console.WriteLine("Remaining {0} records of {1} as the last batch", idx, typeof(AttendeeModel).Name);
                    batchList.Add(batch); // Add remaining batch
                }
            }
            System.Console.WriteLine("{0} records have been loaded into {1} batches", numTotalRecords, batchList.Count);
            return batchList;
        }

        private static AttendeeData ParseLine(string line)
        {
            int[] columnType = new int[] { 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };

            String[] text = new string[columnType.Where(x => x == 0).Count()];
            bool boolVal = false;
            float[] numericalVal = new float[columnType.Where(x => x == 2).Count()];

            int fieldToAnalyzeStart;
            int fId;
            int catId;
            int numericalId;
            bool skipFieldSep;
            int i;
            int coundSeqLen;
            int len;

            int stackTot;
            int stackTop;
            int stackBase;

            fieldToAnalyzeStart = 0;
            fId = 0;
            catId = 0;
            numericalId = 0;
            skipFieldSep = false;
            i = 0;
            coundSeqLen = 0;
            len = line.Length;
            stackTot = 0;
            stackTop = 0;
            stackBase = 0;

            // parse initial '"'
            while (line[coundSeqLen] == '"')
            {
                coundSeqLen++;
            }
            i = coundSeqLen;

            stackTot = stackTop = coundSeqLen;
            stackBase = coundSeqLen > 0 ? 1 : 0;

            while (i < len)
            {
                coundSeqLen = 0;
                while (i + coundSeqLen < len && line[i + coundSeqLen] == '"') coundSeqLen++;
                i += coundSeqLen;

                int prevQoutSeqLen = coundSeqLen;
                if (coundSeqLen > 0)
                {
                    if (coundSeqLen > stackTot)
                    {
                        stackTot |= coundSeqLen;
                        stackTop = coundSeqLen & (~coundSeqLen >> 1);
                    }
                    else if (coundSeqLen == stackTop / 2)
                    {
                        stackTot -= stackTop;
                        stackTot -= coundSeqLen;
                        stackTop = coundSeqLen / 2;
                    }
                    else
                    {
                        stackTot ^= coundSeqLen;
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
                    var startIdx = fieldToAnalyzeStart + prevQoutSeqLen + 1;
                    var length = i - fieldToAnalyzeStart - prevQoutSeqLen * 2 - 1;
                    var token = line.Substring(startIdx, length);

                    switch (columnType[fId])
                    {
                        case (0): // categorical
                                  //int pos = strToIntMaps[catId].Get(text[fId]);
                                  //int offset = srcIdxToCounts[catId];
                            text[catId++] = token;
                            break;
                        case (1): // bool
                            boolVal = ToBool(token);
                            break;
                        case (2): // int
                            numericalVal[numericalId++] = ToInt(token);
                            break;
                        default:
                            throw new Exception("Field type not found");
                    }
                    fId++;
                    fieldToAnalyzeStart = i;
                }

                i++;
            }

            AttendeeData data = new AttendeeData()
            {
                ProgramOwner = text[0],
                Timezone = text[1],
                Timezonemapping = text[2],
                Area = text[3],
                Country = text[4],
                LocalizationLanguage = text[5],
                StartDateDayOfWeek = text[6],
                StartDateHrofDay = numericalVal[1],
                StartDateMonthOfYear = numericalVal[2],
                Duration = numericalVal[3],
                DaysBetweenCreationandWebinar = numericalVal[4],
                RegisteredCount = numericalVal[5],
                Label = numericalVal[0],
                SearchPublishToSearch = boolVal,
                CeOnPremise = numericalVal[6],
                CloudPlatform = numericalVal[7],
                DeveloperTools = numericalVal[8],
                DevicesXbox = numericalVal[9],
                DynamicsCloud = numericalVal[10],
                DynamicsOnPremise = numericalVal[11],
                EnterpriseSolutions = numericalVal[12],
                Hololens = numericalVal[13],
                MicrosoftAzure = numericalVal[14],
                MicrosoftDynamics = numericalVal[15],
                Office = numericalVal[16],
                OtherProductsNotListedAbove = numericalVal[17],
                ProdUncategorized = numericalVal[18],
                ServerManagement = numericalVal[19],
                Skype = numericalVal[20],
                SmallBusinessSolutions = numericalVal[21],
                Surface = numericalVal[22],
                Windows = numericalVal[23],
                WindowsPcTablet = numericalVal[24],
                WindowsEmbeddedIot = numericalVal[25],
                WindowsPhone = numericalVal[26],
                WindowsVolumeLicensing = numericalVal[27],
                BusinessDecisionMaker = numericalVal[28],
                BusinessProfessionals = numericalVal[29],
                Consumers = numericalVal[30],
                Developers = numericalVal[31],
                Educators = numericalVal[32],
                ForHome = numericalVal[33],
                Government = numericalVal[34],
                InformationWorker = numericalVal[35],
                ItDecisionMaker = numericalVal[36],
                ItProfessionals = numericalVal[37],
                OtherAudience = numericalVal[38],
                Partners = numericalVal[39],
                Students = numericalVal[40],
                TAUncategorized = numericalVal[41]
            };
            return data;
        }

        internal static bool ToBool(String text)
        {
            char val = text[0];

            switch (val)
            {
                case 'y':
                    if (text.Length == 3 && text[1] == 'e' && text[2] == 's')
                    {
                        return true;
                    }

                    throw new Exception("A boolean should be either yes or no.");
                case 'n':
                    if (text.Length == 2 && text[1] == 'o')
                    {
                        return false;
                    }

                    throw new Exception("A boolean should be either yes or no.");
                default:
                    throw new Exception("A boolean should be either yes or no.");
            }
        }

        internal static int ToInt(String text)
        {
            int s = 1;
            int i = -1;
            int res = 0;

            try
            {
                if (text[0] == '-')
                {
                    s = -1;
                    i = 0;
                }

                while (++i < text.Length)
                {
                    res = res * 10 + (text[i] - '0');
                }

                return res = res * s;
            }
            catch (Exception)
            {
                throw new Exception("int value malformed");
            }
        }
    }
}
