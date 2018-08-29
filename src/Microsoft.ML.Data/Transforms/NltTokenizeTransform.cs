
#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.Data.Transforms;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.NaturalLanguage.Runtime.Managed.API;

[assembly: LoadableClass(NltTokenizeTransform.Summary, typeof(NltTokenizeTransform), typeof(NltTokenizeTransform.Arguments), typeof(SignatureDataTransform),
    "NLT Tokenizer Transform", "NltTokenizeTransform", "NltToken")]
/*
[assembly: LoadableClass(NltTokenizeTransform.Summary, typeof(NltTokenizeTransform), typeof(NltTokenizeTransform.TokenizeArguments),
    "NLT Tokenizer Transform", "NltTokenizeTransform", "NltToken")]
    */

[assembly: LoadableClass(NltTokenizeTransform.Summary, typeof(NltTokenizeTransform), null, typeof(SignatureLoadDataTransform),
    "NLT Tokenizer Transform", NltTokenizeTransform.LoaderSignature)]

namespace Microsoft.ML.Data.Transforms
{
    /// <summary>
    /// A transform that uses an Office NL Toolkit module to identify token spans in an input string.
    /// The input to this transform is text, and the output is a vector of text containing the words
    /// (tokens) in the original text. The transform can output, optionally, the types of the tokens too.
    /// </summary>
    public sealed class NltTokenizeTransform : RowToRowMapperTransformBase, ITransformTemplate
    {
        internal const string Summary = "A transform that uses an Office NL Toolkit module to identify token spans in an input string. "
            + "The input to this transform is text, and the output is a vector of text containing the words (tokens) in the original text. "
            + "The transform can output, optionally, the types of the tokens too.";

        /// <summary>
        /// Default tokenizer is for whitespace languages only. Here is a brief description of the major differences
        /// between default tokenizer and some language-specific tokenizers:
        /// 1) Initialisms:
        ///     Default - Sequences of single letter followed by a period are one token: E.g., "A." and "A.B.C.".
        ///     English - Same as Default.
        ///     Italian - Same as Default.
        /// 2) Abbreviations:
        ///     Default - English period-ending abbreviations are single tokens: E.g., "cf.".
        ///     English - Same as Default.
        ///     Italian - Italian period-ending abbreviations are single tokens: E.g., "Sig.".
        /// 3) Apostrophe-affix handling:
        ///     Default - E.g., "book’s" and "d’arte" are single tokens; apostrophe characters are handled as a non-breaking character of word.
        ///     English - "book’s" is tokenized into "book" and "’s" and "‘s" has Type=SUFFIX.
        ///     Italian - "d’arte" is tokenized into "d’" and "arte", and "d’" has Type=PREFIX.
        /// 4) Handling of Ordinal numbers:
        ///     Default - E.g., "18th" is tokenized into "18" and "th", and similarly "1o".
        ///     English - "18th" is one token, Type=ORDINAL_NUMBER
        ///     Italian - "1o" is one token, Type=ORDINAL_NUMBER.
        /// 5) Handling of Digits"
        ///     Default - Digit characters immediately preceded by a letter will not be separated from the latter: "Y2K" and "Win10" are single tokens.
        ///             - Digit characters immediately followed by a letter will be separated from the latter: "9am" are tokenized into "9" and "am".
        ///     English - Same as Default.
        ///     Italian - Same as Default.
        ///
        /// This enumeration is serialized.
        /// </summary>
        public enum Language
        {
            Default = 0,
            English = 1,
            French = 2,
            German = 3,
            Dutch = 4,
            Italian = 5,
            Japanese = 6,
            Spanish = 7
        }

        public sealed class Column
        {
            // Input
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the source column", ShortName = "src")]
            public string Source;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Optional column to use for languages. This overrides tokenizer language value.", ShortName = "langscol")]
            public string LanguagesColumn;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Tokenizer language (optional). A Default tokenizer is used that works for white-space languages only.",
                ShortName = "lang")]
            public Language? Language;

            // Output
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the tokens column", ShortName = "tokenscol")]
            public string TokensColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Optional column to use for tokens types.", ShortName = "typescol")]
            public string TypesColumn;

            /// <summary>
            /// For parsing Column from a string. This supports "source" or "tokensColumn:source".
            /// </summary>
            public static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private bool TryParse(string str)
            {
                return ColumnParsingUtils.TryParse(str, out TokensColumn, out Source);
            }

            /// <summary>
            /// The unparsing functionality, for generating succinct command line forms "source" or "tokensColumn:source".
            /// </summary>
            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (!TrySanitize() || Language.HasValue || !string.IsNullOrWhiteSpace(LanguagesColumn)
                    || !string.IsNullOrWhiteSpace(TypesColumn))
                {
                    return false;
                }

                sb.Append(TokensColumn).Append(':').Append(Source);
                return true;
            }

            /// <summary>
            /// If both of tokensColumn and source are null or white-space, return false.
            /// Otherwise, if one is null or white-space, assign that one the other's value.
            /// </summary>
            public bool TrySanitize()
            {
                if (string.IsNullOrWhiteSpace(TokensColumn))
                    TokensColumn = Source;
                else if (string.IsNullOrWhiteSpace(Source))
                    Source = TokensColumn;
                return !string.IsNullOrWhiteSpace(TokensColumn);
            }
        }

        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Optional column to use for languages. This overrides tokenizer language value.",
                ShortName = "langscol", SortOrder = 1,
                Purpose = SpecialPurpose.ColumnName)]
            public string LanguagesColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Tokenizer Language.", ShortName = "lang", SortOrder = 1)]
            public Language Language;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Create a separate column for tokens types.", ShortName = "types", SortOrder = 1)]
            public bool CreateTypesColumn;
        }

        public sealed class Arguments : ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "New column definition(s)", ShortName = "col", SortOrder = 1)]
            public Column[] Column;
        }

        public sealed class TokenizeArguments : ArgumentsBase
        {
        }

        private sealed class Bindings : ColumnBindingsBase
        {
            // We may have one or more source columns. Each source column has one or two corresponding output
            // columns. ColGroupInfo represents a group of output columns associated with a single source column.
            // The count of ColGroupInfo(s) equals to the source columns count.
            public sealed class ColGroupInfo
            {
                public readonly Language Lang;
                public readonly int SrcColIndex;
                public readonly ColumnType SrcColType;
                public readonly int LangsColIndex;

                // If tokens column, are we outputting token types too?
                public readonly bool RequireTypes;

                internal readonly string SrcColName;
                internal readonly string LangsColName;

                public ColGroupInfo(Language lang, int srcColIndex, string srcColName, ColumnType srcColType,
                    int langsColIndex, string langsColName, bool requireTypes = false)
                {
                    Contracts.Assert(Enum.IsDefined(typeof(Language), lang));
                    Contracts.Assert(srcColIndex >= 0);
                    Contracts.AssertNonEmpty(srcColName);
                    Contracts.Assert(srcColType.ItemType.IsText);
                    Contracts.Assert((langsColIndex >= 0 && langsColName != null)
                        || (langsColIndex == -1 && langsColName == null));

                    Lang = lang;
                    SrcColIndex = srcColIndex;
                    SrcColName = srcColName;
                    SrcColType = srcColType;
                    LangsColIndex = langsColIndex;
                    LangsColName = langsColName;
                    RequireTypes = requireTypes;
                }
            }

            public sealed class ColInfo
            {
                public readonly int GroupInfoId;
                public readonly bool IsTypes;

                public ColInfo(int groupInfoId, bool isTypes = false)
                {
                    Contracts.Assert(groupInfoId >= 0);

                    GroupInfoId = groupInfoId;
                    IsTypes = isTypes;
                }
            }

            private const string LangTypeName = "text";
            private const string SrcTypeName = "scalar/vector of text";

            private readonly ColumnType _outColType;
            private readonly string[] _names;
            private readonly bool _user;

            public readonly ColInfo[] Infos;
            public readonly ColGroupInfo[] GroupInfos;

            public int SrcColsCount
            {
                get
                {
                    Contracts.AssertNonEmpty(GroupInfos);
                    return GroupInfos.Length;
                }
            }

            private Bindings(ColGroupInfo[] groups, ColInfo[] infos, ISchema input, bool user, string[] names)
                : base(input, user, names)
            {
                Contracts.Assert(InfoCount > 0);
                Contracts.Assert(Utils.Size(names) == InfoCount);
                Contracts.Assert(Utils.Size(infos) == InfoCount);
                Contracts.AssertNonEmpty(groups);

                Infos = infos;
                GroupInfos = groups;
                _outColType = new VectorType(TextType.Instance);
                _names = names;
                _user = user;
            }

            public static Bindings Create(Bindings prev, ISchema input)
            {
                int groupsLen = prev.GroupInfos.Length;
                var groups = new ColGroupInfo[groupsLen];
                for (int i = 0; i < groupsLen; i++)
                {
                    var groupInfo = prev.GroupInfos[i];
                    var lang = groupInfo.Lang;
                    var srcName = groupInfo.SrcColName;
                    int srcIdx;
                    ColumnType srcType;
                    Bind(input, srcName, t => t.ItemType.IsText, SrcTypeName, out srcIdx, out srcType, false);

                    var langsName = groupInfo.LangsColName;
                    int langsIdx;
                    if (langsName != null)
                    {
                        ColumnType langsType;
                        Bind(input, langsName, t => t.IsText, LangTypeName, out langsIdx, out langsType, false);
                    }
                    else
                        langsIdx = -1;

                    bool requireTypes = groupInfo.RequireTypes;
                    groups[i] = new ColGroupInfo(lang, srcIdx, srcName, srcType, langsIdx, langsName, requireTypes);

                }

                return new Bindings(groups, prev.Infos.ToArray(), input, false, prev._names);

            }

            public static Bindings Create(Arguments args, ISchema input, IChannel ch, bool exceptUser = true)
            {
                Contracts.AssertValue(ch);
                ch.AssertValue(args);
                ch.CheckUserArg(Utils.Size(args.Column) > 0, nameof(args.Column), "Column is missing");
                ch.CheckUserArg(Enum.IsDefined(typeof(Language), args.Language),
                    nameof(args.Language), "'lang' has invalid value");

                int namesLen = 0;
                foreach (var col in args.Column)
                {
                    ch.CheckUserArg(!string.IsNullOrWhiteSpace(col.Source), nameof(col.Source), "Cannot be empty");
                    ch.CheckUserArg(!string.IsNullOrWhiteSpace(col.TokensColumn), nameof(col.TokensColumn), "Cannot be empty");
                    ch.CheckUserArg(!col.Language.HasValue || Enum.IsDefined(typeof(Language), col.Language),
                        nameof(col.Language), "Value is invalid");

                    namesLen += string.IsNullOrWhiteSpace(col.TypesColumn) && !args.CreateTypesColumn ? 1 : 2;
                }

                int index = 0;
                var names = new string[namesLen];
                var infos = new ColInfo[names.Length];
                var groups = new ColGroupInfo[args.Column.Length];
                for (int i = 0; i < args.Column.Length; i++)
                {
                    var col = args.Column[i];
                    int srcIdx;
                    ColumnType srcType;
                    Bind(input, col.Source, t => t.ItemType.IsText, SrcTypeName, out srcIdx, out srcType, exceptUser);
                    ch.Assert(srcIdx >= 0);

                    int langsColIdx;
                    string langsCol = !string.IsNullOrWhiteSpace(col.LanguagesColumn)
                        ? col.LanguagesColumn
                        : args.LanguagesColumn;
                    if (!string.IsNullOrWhiteSpace(langsCol))
                    {
                        ColumnType langsColType;
                        Bind(input, langsCol, t => t.IsText, LangTypeName, out langsColIdx, out langsColType, exceptUser);
                        ch.Assert(langsColIdx >= 0);
                    }
                    else
                    {
                        langsCol = null;
                        langsColIdx = -1;
                    }

                    var lang = col.Language ?? args.Language;
                    bool requireTypes = !string.IsNullOrWhiteSpace(col.TypesColumn) || args.CreateTypesColumn;

                    groups[i] = new ColGroupInfo(lang, srcIdx, col.Source, srcType, langsColIdx, langsCol, requireTypes);
                    names[index] = col.TokensColumn;
                    infos[index++] = new ColInfo(i);
                    if (requireTypes)
                    {
                        names[index] = !string.IsNullOrWhiteSpace(col.TypesColumn) ? col.TypesColumn
                            : string.Format("{0}_Tokens", col.Source);
                        infos[index++] = new ColInfo(i, isTypes: true);
                    }
                }
                ch.Assert(index == namesLen);

                return new Bindings(groups, infos.ToArray(), input, exceptUser, names.ToArray());
            }

            /// <summary>
            /// Binds and validate the type of a column with the given name using input schema and returns the column index.
            /// Fails if there is no column with the given name or if the column type is not valid.
            /// </summary>
            /// <param name="input">The input Schema.</param>
            /// <param name="name">The column name.</param>
            /// <param name="isValidType">Whether the column type is valid.</param>
            /// <param name="expectedType">The expected type of the column.</param>
            /// <param name="index">The column index.</param>
            /// <param name="type">The column type.</param>
            /// <param name="exceptUser">A flag that determines the exception type thrown in failure.</param>
            private static void Bind(ISchema input, string name, Predicate<ColumnType> isValidType, string expectedType,
                out int index, out ColumnType type, bool exceptUser = true)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(name);

                if (name == "text")
                {
                    name = "Text";
                }
                if (!input.TryGetColumnIndex(name, out index))
                {
                    throw exceptUser
                        ? Contracts.ExceptUserArg(nameof(Arguments.Column), "Source column '{0}' not found", name)
                        : Contracts.ExceptDecode("Source column '{0}' not found", name);
                }

                type = input.GetColumnType(index);
                if (!isValidType(type))
                {
                    throw exceptUser
                        ? Contracts.ExceptUserArg(nameof(Arguments.Column), "Source column '{0}' has type '{1}' but must be '{2}'", name, type, expectedType)
                        : Contracts.ExceptDecode("Source column '{0}' has type '{1}' but must be '{2}'", name, type, expectedType);
                }
            }

            public static Bindings Create(ModelLoadContext ctx, ISchema input, IChannel ch)
            {
                Contracts.AssertValue(ch);
                ch.AssertValue(ctx);

                // *** Binary format ***
                // int: count of group column infos (ie, count of source columns)
                // For each group column info
                //     int: the tokenizer language
                //     int: the id of source column name
                //     int: the id of languages column name
                //     bool: whether the types output is required
                //     For each column info that belongs to this group column info
                //     (either one column info for tokens or two for tokens and types)
                //          int: the id of the column name

                int groupsLen = ctx.Reader.ReadInt32();
                ch.CheckDecode(groupsLen > 0);

                var names = new List<string>();
                var infos = new List<ColInfo>();
                var groups = new ColGroupInfo[groupsLen];
                for (int i = 0; i < groups.Length; i++)
                {
                    int lang = ctx.Reader.ReadInt32();
                    ch.CheckDecode(Enum.IsDefined(typeof(Language), lang));

                    string srcName = ctx.LoadNonEmptyString();
                    int srcIdx;
                    ColumnType srcType;
                    Bind(input, srcName, t => t.ItemType.IsText, SrcTypeName, out srcIdx, out srcType, false);

                    string langsName = ctx.LoadStringOrNull();
                    int langsIdx;
                    if (langsName != null)
                    {
                        ColumnType langsType;
                        Bind(input, langsName, t => t.IsText, LangTypeName, out langsIdx, out langsType, false);
                    }
                    else
                        langsIdx = -1;

                    bool requireTypes = ctx.Reader.ReadBoolByte();
                    groups[i] = new ColGroupInfo((Language)lang, srcIdx, srcName, srcType, langsIdx, langsName, requireTypes);

                    infos.Add(new ColInfo(i));
                    names.Add(ctx.LoadNonEmptyString());
                    if (requireTypes)
                    {
                        infos.Add(new ColInfo(i, isTypes: true));
                        names.Add(ctx.LoadNonEmptyString());
                    }
                }

                return new Bindings(groups, infos.ToArray(), input, false, names.ToArray());
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: count of group column infos (ie, count of source columns)
                // For each group column info
                //     int: the tokenizer language
                //     int: the id of source column name
                //     int: the id of languages column name
                //     bool: whether the types output is required
                //     For each column info that belongs to this group column info
                //     (either one column info for tokens or two for tokens and types)
                //          int: the id of the column name

                Contracts.Assert(Utils.Size(GroupInfos) > 0);
                ctx.Writer.Write(GroupInfos.Length);

                int iinfo = 0;
                for (int groupId = 0; groupId < GroupInfos.Length; groupId++)
                {
                    var groupInfo = GroupInfos[groupId];

                    Contracts.Assert(Enum.IsDefined(typeof(Language), groupInfo.Lang));
                    ctx.Writer.Write((int)groupInfo.Lang);
                    ctx.SaveNonEmptyString(groupInfo.SrcColName);
                    ctx.SaveStringOrNull(groupInfo.LangsColName);
                    ctx.Writer.WriteBoolByte(groupInfo.RequireTypes);

                    int count = groupInfo.RequireTypes ? 2 : 1;
                    int lim = iinfo + count;
                    for (; iinfo < lim; iinfo++)
                    {
                        Contracts.Assert(Infos[iinfo].GroupInfoId == groupId);
                        ctx.SaveNonEmptyString(GetColumnNameCore(iinfo));
                    }
                }
                Contracts.Assert(iinfo == Infos.Length);
            }

            protected override ColumnType GetColumnTypeCore(int iinfo)
            {
                Contracts.Assert(0 <= iinfo && iinfo < Infos.Length);
                return _outColType;
            }

            internal Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = GetActiveInput(predicate);
                for (int i = 0; i < Infos.Length; i++)
                {
                    if (predicate(MapIinfoToCol(i)))
                    {
                        var groupInfo = GroupInfos[Infos[i].GroupInfoId];
                        active[groupInfo.SrcColIndex] = true;
                        if (groupInfo.LangsColIndex != -1)
                            active[groupInfo.LangsColIndex] = true;
                    }
                }

                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        public const string LoaderSignature = "NltTokenizeTransform";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "NLTTOKNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private readonly Bindings _bindings;

        // An array of tokenizers indexed by tokenizer language id
        // REVIEW nihejazi: Dispose() method of these tokenizers will not be called.
        private static volatile Tokenizer[] _toks;

        private const string RegistrationName = "NltTokenize";
        private const string TokenizersDirectoryName = "Tokenizers";

        private static Tokenizer[] Tokenizers
        {
            get
            {
                if (_toks == null)
                {
                    var values = Enum.GetValues(typeof(Language)).Cast<int>();
                    int maxValue = values.Max();
                    Contracts.Assert(values.Min() >= 0);
                    Contracts.Assert(maxValue < Utils.ArrayMaxSize);
                    var toks = new Tokenizer[maxValue + 1];
                    Interlocked.CompareExchange(ref _toks, toks, null);
                }

                return _toks;
            }
        }

        public NltTokenizeTransform(IHostEnvironment env, Arguments args, RoleMappedData data)
            : this(env, args, Contracts.CheckRef(data, nameof(data)).Data)
        {
        }

        /// <summary>
        /// Public constructor corresponding to SignatureDataTransform.
        /// </summary>
        public NltTokenizeTransform(IHostEnvironment env, Arguments args, IDataView input)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));

            using (var ch = Host.Start("Construction"))
            {
                _bindings = Bindings.Create(args, Source.Schema, ch);
                CheckResources(ch);
                ch.Done();
            }
        }

        /// <summary>
        /// Public constructor corresponding to SignatureTokenizeTransform. It accepts arguments of type ArgumentsBase,
        /// and a separate array of columns (constructed from the caller -WordBag/WordHashBag- arguments).
        /// </summary>
        public NltTokenizeTransform(IHostEnvironment env, TokenizeArguments args, IDataView input, OneToOneColumn[] columns)
            : base(env, RegistrationName, input)
        {
            Host.CheckValue(args, nameof(args));
            Host.CheckUserArg(Utils.Size(columns) > 0, nameof(Arguments.Column));

            var tokenizerColumns = new Column[columns.Length];
            for (int i = 0; i < columns.Length; i++)
            {
                Host.CheckUserArg(!string.IsNullOrWhiteSpace(columns[i].Name), nameof(OneToOneColumn.Name));
                Host.CheckUserArg(!string.IsNullOrWhiteSpace(columns[i].Source), nameof(OneToOneColumn.Source));
                tokenizerColumns[i] = new Column { Source = columns[i].Source, TokensColumn = columns[i].Name };
            }

            var extendedArgs = new Arguments
            {
                Column = tokenizerColumns,
                Language = args.Language,
                LanguagesColumn = args.LanguagesColumn,
                CreateTypesColumn = args.CreateTypesColumn
            };

            using (var ch = Host.Start("Construction"))
            {
                _bindings = Bindings.Create(extendedArgs, Source.Schema, ch);
                CheckResources(ch);
                ch.Done();
            }
        }

        private NltTokenizeTransform(IHost host, ModelLoadContext ctx, IDataView input)
            : base(host, input)
        {
            Host.AssertValue(ctx);

            using (var ch = Host.Start("Deserialization"))
            {
                // *** Binary format ***
                // bindings
                _bindings = Bindings.Create(ctx, Source.Schema, ch);

                CheckResources(ch);
                ch.Done();
            }
        }

        private NltTokenizeTransform(IHostEnvironment env, NltTokenizeTransform xf, IDataView newSource)
            : base(env, RegistrationName, newSource)
        {
            _bindings = Bindings.Create(xf._bindings, newSource.Schema);
        }

        private void CheckResources(IChannel ch)
        {
            Contracts.AssertValue(ch);

            // Find required resources
            var requiredResources = new bool[Tokenizers.Length];
            foreach (var group in _bindings.GroupInfos)
                requiredResources[(int)group.Lang] = true;

            // Check the existence of resource files
            var missings = new StringBuilder();
            foreach (Language lang in Enum.GetValues(typeof(Language)))
            {
                if (GetResourceFilePathOrNull(lang) == null)
                {
                    if (requiredResources[(int)lang])
                    {
                        throw Contracts.Except(
                            "Missing '{0}.tok' resource.",
                            lang); // , ResourceConfigurations.HelpPage);
                    }

                    if (missings.Length > 0)
                        missings.Append(", ");
                    missings.Append(lang);
                }
            }

            if (missings.Length > 0)
            {
                const string wrnMsg = "Missing resources for languages: '{0}'."
                    + "Default tokenizer (specified by 'lang' option) will be used if needed.";
                ch.Warning(wrnMsg, missings.ToString()); // , ResourceConfigurations.HelpPage);
            }
        }

        public static NltTokenizeTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(ctx, nameof(ctx));
            h.CheckValue(input, nameof(input));
            ctx.CheckAtModel(GetVersionInfo());
            return h.Apply("Loading Model", ch => new NltTokenizeTransform(h, ctx, input));
        }

        public override ISchema Schema { get { return _bindings; } }

        public override void Save(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // bindings
            _bindings.Save(ctx);
        }

        private static string GetResourceFilePathOrNull(Language lang)
        {
            return Utils.FindExistentFileOrNull(lang + ".tok", TokenizersDirectoryName, assemblyForBasePath: typeof(NltTokenizeTransform));
        }

        protected override bool? ShouldUseParallelCursors(Func<int, bool> predicate)
        {
            Host.AssertValue(predicate, "predicate");

            // Prefer parallel cursors iff some of our columns are active, otherwise, don't care.
            if (_bindings.AnyNewColumnsActive(predicate))
                return true;
            return null;
        }

        protected override IRowCursor GetRowCursorCore(Func<int, bool> predicate, IRandom rand = null)
        {
            Host.AssertValue(predicate, "predicate");
            Host.AssertValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var input = Source.GetRowCursor(inputPred, rand);
            return new RowCursor(this, input, active);
        }

        public override IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> predicate, int n, IRandom rand = null)
        {
            Host.CheckValue(predicate, nameof(predicate));
            Host.CheckValueOrNull(rand);

            var inputPred = _bindings.GetDependencies(predicate);
            var active = _bindings.GetActive(predicate);
            var inputs = Source.GetRowCursorSet(out consolidator, inputPred, n, rand);
            Host.AssertNonEmpty(inputs);

            if (inputs.Length == 1 && n > 1 && _bindings.AnyNewColumnsActive(predicate))
                inputs = DataViewUtils.CreateSplitCursors(out consolidator, Host, inputs[0], n);
            Host.AssertNonEmpty(inputs);

            var cursors = new IRowCursor[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                cursors[i] = new RowCursor(this, inputs[i], active);
            return cursors;
        }

        protected override Func<int, bool> GetDependenciesCore(Func<int, bool> predicate)
        {
            return _bindings.GetDependencies(predicate);
        }

        protected override Delegate[] CreateGetters(IRow input, Func<int, bool> active, out Action disp)
        {
            Func<int, bool> activeInfos =
                iinfo =>
                {
                    int col = _bindings.MapIinfoToCol(iinfo);
                    return active(col);
                };

            disp = null;

            var getters = new Delegate[_bindings.InfoCount];
            var groupCurHelpers = new GroupCursorHelper[_bindings.SrcColsCount];
            var resourcesExist = new bool?[Tokenizers.Length];
            for (int iinfo = 0; iinfo < _bindings.Infos.Length; iinfo++)
            {
                if (!activeInfos(iinfo))
                    continue;

                int groupId = _bindings.Infos[iinfo].GroupInfoId;
                if (groupCurHelpers[groupId] == null)
                    groupCurHelpers[groupId] = new GroupCursorHelper(input, _bindings.GroupInfos[groupId], resourcesExist);
                var groupState = groupCurHelpers[groupId];
                ValueGetter<VBuffer<DvText>> fn;
                if (_bindings.Infos[iinfo].IsTypes)
                    fn = (ref VBuffer<DvText> dst) => groupState.GetTypes(ref dst);
                else
                    fn = (ref VBuffer<DvText> dst) => groupState.GetTokens(ref dst);
                getters[iinfo] = fn;
            }
            return getters;
        }

        protected override int MapColumnIndex(out bool isSrc, int col)
        {
            return _bindings.MapColumnIndex(out isSrc, col);
        }

        public IDataTransform ApplyToData(IHostEnvironment env, IDataView newSource)
        {
            return new NltTokenizeTransform(env, this, newSource);
        }

        public Dictionary<string, object> GetParams()
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();

            // TODO Need a systematic way to identify whether this transform is the first one.
            Host.Assert(_bindings.GroupInfos.Length == 1);
            var groupInfo = _bindings.GroupInfos[0];
            var lang = groupInfo.Lang;
            var langIdx = groupInfo.LangsColIndex;
            var langsColName = groupInfo.LangsColName;
            var srcColIdx = groupInfo.SrcColIndex;

            parameters.Add("srcColIdx", srcColIdx);
            // Others go this way as well.
            return parameters;
        }

        public void WriteParams(BinaryWriter writer)
        {
            throw new NotImplementedException();
        }

        private sealed class GroupCursorHelper
        {
            private readonly IRow _source;
            private readonly Bindings.ColGroupInfo _groupInfo;
            private readonly ValueGetter<DvText> _langGetter;
            private readonly Action _ensureTokensAndTypesDel;
            private readonly bool?[] _resourcesExist;

            private readonly List<DvText> _tokensList;
            private readonly List<DvText> _typesList;

            private VBuffer<DvText> _tokensBuffer;
            private VBuffer<DvText> _typesBuffer;

            // The counters track the row corresponding to the values in the tokens/types buffer.
            private long _position;

            private static volatile Dictionary<DvText, Language> _langsDictionary;

            private static Dictionary<DvText, Language> LangsDictionary
            {
                get
                {
                    if (_langsDictionary == null)
                    {
                        var langsDictionary = Enum.GetValues(typeof(Language)).Cast<Language>()
                            .ToDictionary(lang => new DvText(lang.ToString()));
                        Interlocked.CompareExchange(ref _langsDictionary, langsDictionary, null);
                    }

                    return _langsDictionary;
                }
            }

            public GroupCursorHelper(IRow input, Bindings.ColGroupInfo groupInfo, bool?[] resourcesExist)
            {
                Contracts.AssertValue(input);
                Contracts.AssertValue(groupInfo);
                _source = input;
                _groupInfo = groupInfo;
                _resourcesExist = resourcesExist;

                if (_groupInfo.LangsColIndex >= 0)
                    _langGetter = _source.GetGetter<DvText>(_groupInfo.LangsColIndex);
                _ensureTokensAndTypesDel = GetEnsureTokensAndTypesDel();
                _tokensList = new List<DvText>();
                if (_groupInfo.RequireTypes)
                    _typesList = new List<DvText>();
                _position = -1;
            }

            public void GetTokens(ref VBuffer<DvText> dst)
            {
                _ensureTokensAndTypesDel();
                _tokensBuffer.CopyTo(ref dst);
            }

            public void GetTypes(ref VBuffer<DvText> dst)
            {
                _ensureTokensAndTypesDel();
                _typesBuffer.CopyTo(ref dst);
            }

            /// <summary>
            /// Returns a delegate that ensures tokens (and types if required) are computed for the
            /// current input position and copied into _tokensBuffer/_typesBuffer.
            /// </summary>
            private Action GetEnsureTokensAndTypesDel()
            {
                if (_groupInfo.SrcColType.IsVector)
                    return GetEnsureTokensAndTypesVecDel();
                return GetEnsureTokensAndTypesOneDel();
            }

            private void AddTokenAndType(string text, int startIndex, int length, string type)
            {
                Contracts.AssertValue(_tokensList);
                if (length > 0)
                {
                    int lim = startIndex + length;
                    var token = new DvText(text, startIndex, lim);
                    _tokensList.Add(token);
                    if (_typesList != null)
                        _typesList.Add(new DvText(type));
                }
            }

            private Action GetEnsureTokensAndTypesOneDel()
            {
                Contracts.Assert(!_groupInfo.SrcColType.IsVector);

                var tokLang = _groupInfo.Lang;
                var lang = default(DvText);
                var src = default(DvText);
                var getSrc = _source.GetGetter<DvText>(_groupInfo.SrcColIndex);
                GetSpansSimpleCallback addTokenAndTypeCallback = AddTokenAndType;
                return
                    () =>
                    {
                        Contracts.Assert(_source.Position >= 0);
                        Contracts.Assert(_position <= _source.Position);

                        if (_position < _source.Position)
                        {
                            var langToUse = tokLang;
                            UpdateLanguage(ref langToUse, ref lang);
                            getSrc(ref src);
                            ClearLists();
                            Tokenize(ref src, langToUse, addTokenAndTypeCallback);
                            CopyLists();
                            _position = _source.Position;
                        }
                    };
            }

            private Action GetEnsureTokensAndTypesVecDel()
            {
                Contracts.Assert(_groupInfo.SrcColType.IsVector);

                var tokLang = _groupInfo.Lang;
                var lang = default(DvText);
                var srcVec = default(VBuffer<DvText>);
                var getSrcVec = _source.GetGetter<VBuffer<DvText>>(_groupInfo.SrcColIndex);
                GetSpansSimpleCallback addTokenAndTypeCallback = AddTokenAndType;
                return
                    () =>
                    {
                        Contracts.Assert(_source.Position >= 0);
                        Contracts.Assert(_position <= _source.Position);

                        if (_position < _source.Position)
                        {
                            var langToUse = tokLang;
                            UpdateLanguage(ref langToUse, ref lang);
                            getSrcVec(ref srcVec);
                            ClearLists();
                            for (int i = 0; i < srcVec.Count; i++)
                                Tokenize(ref srcVec.Values[i], langToUse, addTokenAndTypeCallback);
                            CopyLists();
                            _position = _source.Position;
                        }
                    };
            }

            private void UpdateLanguage(ref Language langToUse, ref DvText langTxt)
            {
                if (_langGetter != null)
                {
                    _langGetter(ref langTxt);
                    Language lang;
                    if (!langTxt.IsNA && LangsDictionary.TryGetValue(langTxt, out lang))
                        langToUse = lang;
                }

                if (!ResourceExists(langToUse))
                    langToUse = Language.Default;
                AddResourceIfNotPresent(langToUse);
            }

            private void ClearLists()
            {
                _tokensList.Clear();
                if (_typesList != null)
                    _typesList.Clear();
            }

            private void CopyLists()
            {
                VBufferUtils.Copy(_tokensList, ref _tokensBuffer, _tokensList.Count);
                if (_typesList != null)
                    VBufferUtils.Copy(_typesList, ref _typesBuffer, _typesList.Count);
            }

            /// <summary>
            /// Add tokens (and types if required) that are computed for src to _tokensList/_typesList.
            /// </summary>
            private void Tokenize(ref DvText src, Language lang, GetSpansSimpleCallback addTokenAndType)
            {
                Contracts.Assert(_typesList == null || _tokensList.Count == _typesList.Count);

                if (!src.HasChars)
                    return;

                int ichMin;
                int ichLim;
                string text = src.GetRawUnderlyingBufferInfo(out ichMin, out ichLim);
                Tokenizers[(int)lang].GetSpansSimple(text, ichMin, ichLim - ichMin, addTokenAndType);
                Contracts.Assert(_typesList == null || _tokensList.Count == _typesList.Count);
            }

            private bool ResourceExists(Language lang)
            {
                int langVal = (int)lang;
                Contracts.Assert(0 <= langVal & langVal < Utils.Size(Tokenizers));
                return Tokenizers[langVal] != null ||
                    (_resourcesExist[langVal] ?? (_resourcesExist[langVal] = GetResourceFilePathOrNull(lang) != null).Value);
            }

            private static void AddResourceIfNotPresent(Language lang)
            {
                Contracts.Assert(0 <= (int)lang & (int)lang < Utils.Size(Tokenizers));

                if (Tokenizers[(int)lang] == null)
                {
                    string tokPath = GetResourceFilePathOrNull(lang);
                    Contracts.Assert(tokPath != null);
                    var tok = new Tokenizer(ResourceLoader.FromNonMemoryMappedFile(tokPath));
                    Interlocked.CompareExchange(ref Tokenizers[(int)lang], tok, null);
                }
            }
        }

        private sealed class RowCursor : SynchronizedCursorBase<IRowCursor>, IRowCursor
        {
            private readonly Bindings _bindings;
            private readonly ValueGetter<VBuffer<DvText>>[] _getters;
            private readonly bool[] _active;

            public RowCursor(NltTokenizeTransform parent, IRowCursor input, bool[] active)
                : base(parent.Host, input)
            {
                Ch.AssertValue(parent._bindings);
                Ch.AssertNonEmpty(parent._bindings.Infos);
                Ch.AssertNonEmpty(parent._bindings.GroupInfos);
                Ch.Assert(active == null || active.Length == parent._bindings.ColumnCount);

                _bindings = parent._bindings;
                _active = active;
                var resourcesExist = new bool?[Tokenizers.Length];

                var infos = parent._bindings.Infos;
                _getters = new ValueGetter<VBuffer<DvText>>[infos.Length];
                var groupCurHelpers = new GroupCursorHelper[_bindings.SrcColsCount];
                for (int iinfo = 0; iinfo < infos.Length; iinfo++)
                {
                    if (_active != null && !_active[_bindings.MapIinfoToCol(iinfo)])
                        continue;

                    int groupId = infos[iinfo].GroupInfoId;
                    if (groupCurHelpers[groupId] == null)
                        groupCurHelpers[groupId] = new GroupCursorHelper(input, _bindings.GroupInfos[groupId], resourcesExist);
                    var groupState = groupCurHelpers[groupId];
                    if (infos[iinfo].IsTypes)
                        _getters[iinfo] = groupState.GetTypes;
                    else
                        _getters[iinfo] = groupState.GetTokens;
                }
            }

            public ISchema Schema { get { return _bindings; } }

            public bool IsColumnActive(int col)
            {
                Ch.Check(0 <= col && col < _bindings.ColumnCount);
                return _active == null || _active[col];
            }

            public ValueGetter<TValue> GetGetter<TValue>(int col)
            {
                Ch.Check(IsColumnActive(col));

                bool isSrc;
                int index = _bindings.MapColumnIndex(out isSrc, col);
                if (isSrc)
                    return Input.GetGetter<TValue>(index);

                Ch.Assert(_getters[index] != null);
                var fn = _getters[index] as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }
        }
    }
}
