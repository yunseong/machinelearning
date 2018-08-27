using System.Runtime.CompilerServices;

namespace Microsoft.ML.Runtime.Tools.Console
{
    public sealed class Functions
    {
        /// <summary>
        /// Returns the correct normalizer scale
        /// </summary>
        internal static float ComputeNormScale(float l2Norm)
        {
            float normScale = 1;
            if (l2Norm > 0)
            {
                normScale /= l2Norm;
            }

            // Don't normalize small values.
            if (normScale < (float)1e-8)
            {
                normScale = 1;
            }
            return normScale;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsPow2(int val)
        {
            return ((val) & ((val) - 1)) == 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsContigBitmask(int val)
        {
            return IsPow2((val) & ((~(val)) << 1));
        }
    }
}
