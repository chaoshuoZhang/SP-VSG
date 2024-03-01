# Self-Prompting Vectorized Sketch Generation with Progressive Diffusion Process(SP-VSG)
We have publicly released our project code here, and the subsequent content will be continuously improved and updated. Thanks to [OpenAI](https://github.com/openai/guided-diffusion) and [SketchKnitter](https://github.com/wangqiang9/SketchKnitter) for the open-source code
# motivation
  Recently, in the field of vector sketch generation, the diffusion model represented by [SketchKnitter](https://github.com/wangqiang9/SketchKnitter) has completely outperformed traditional iterative stroke point generation methods represented by [sketchRNN](https://magenta.tensorflow.org/sketch_rnn) in terms of image quality. Building upon the global modeling of example sketches by SketchKnitter, particularly emphasizing recognizability embedded during the diffusion backward process, and inspired by another contemporaneous work, AdamDiff, we further explored the variations in recognizability during the backward propagation process for sketches of different categories and sketches of the same category but with different stroke point lengths. The results demonstrate that targets with simpler sketch structures exhibit faster improvement in recognizability during the backward process. Based on this observation, we realized that leveraging the differences in recognizability changes during the backward process for sketches with varying structural complexities could enhance the generation of complex vector sketches. This approach resembles an apprentice imitating sketches to improve the quality of their own work. We refer to this as self-guided vector sketch generation.

  <img src="images/fig_1.jpg" alt="Comparison of Different Method Architectures." width="400">

  <img src="images/identifiability_comparison.png" alt="N represents the length of the vector sketch, while Steps denote the time steps in the diffusion model's backward process." width="600">

# Method
# result
