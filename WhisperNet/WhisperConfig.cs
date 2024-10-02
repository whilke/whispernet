using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

namespace WhisperNet
{
public class WhisperConfig
    {
        // default props

        public int min_length = 0;
        // Max length per inference
        public int max_length = 448;
        public float repetition_penalty = 1.0f;
        public int num_beams = 1;
        public int num_return_sequences = 1;
        public float length_penalty = 1.0f;



        public SessionOptions GetSessionOptionsForEp()
        {
            var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CPU();
            return sessionOptions;
        }
    }
}
