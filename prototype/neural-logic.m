(* ::Package:: *)

(* 
  TODO:
  - Bias term
  - Initialisation policies
  - Work out the policy for the sizes that ensure all possible DNF expressions
    can be learned. Ten change parameter to [0, 1] from 0 capacity to full capacity to represent
    all possible DNF expressions. Ignoring NOTs then andSize does not need to be larger than 2^(inputSize - 1)
*)

(* ------------------------------------------------------------------ *)
BeginPackage["neurallogic`"]
(* ------------------------------------------------------------------ *)

Harden::usage = "Hards soft bits.";
Soften::usage = "Soften hard bits.";
NeuralNOT::usage = "Neural NOT.";
HardNeuralAND::usage = "Hard neural AND.";
HardNeuralNAND::usage = "Hard neural NAND.";
HardNeuralOR::usage = "Hard neural OR.";
HardNeuralNOR::usage = "Hard neural NOR.";
PortLayer::usage = "Port layer.";
AppendHardClassificationLoss::usage = "Append hard classification loss.";
NeuralOR::usage = "Neural OR.";
NeuralAND::usage = "Neural AND.";
HardClip::usage = "Hard clip.";
LogisticClip::usage = "Logistic clip.";
HardNOT::usage = "Hard NOT.";
HardNeuralMajority::usage = "Hard neural majority.";

(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)
Begin["`Private`"]
(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)
(* Boolean utilities *)
(* ------------------------------------------------------------------ *)

Harden[softBit_] := If[softBit > 0.5, True, False]
Harden[softBits_List] := Map[Harden, softBits]

Soften[hardBit_] := If[hardBit == 1, 1.0, 0.0]
Soften[hardBits_List] := Map[Soften, hardBits]

HardClip[x_] := Clip[x, {0.00000001, 0.9999999}]
LogisticClip[x_] := LogisticSigmoid[4(2 x - 1)]

(* ------------------------------------------------------------------ *)
(* Differentiable HARD NOT, HARD AND, HARD OR *)
(* ------------------------------------------------------------------ *)

HardNOT[input_, weights_] := 1 - weights + input (2 weights - 1)

InitializeBalanced[net_] := NetInitialize[net,
  Method -> {"Random", "Weights" -> UniformDistribution[{0.4, 0.6}]}
]

InitializeBiasToZero[net_] := NetInitialize[net,
  Method -> {"Random", "Weights" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[-1, 1]]}
]

InitializeBiasToOne[net_] := NetInitialize[net,
  Method -> {"Random", "Weights" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[1, 1]]}
]

NeuralNOT[inputSize_] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Output" -> inputSize],
    "WeightsClip" -> ElementwiseLayer[HardClip], 
    "Not" -> ThreadingLayer[HardNOT[#Input, #Weights] &, 1],
    "OutputClip" -> ElementwiseLayer[LogisticClip] 
  |>,
  {
    "Weights" -> "WeightsClip",
    "WeightsClip" -> NetPort["Not", "Weights"],
    "Not" -> "OutputClip"
  }
]

HardAND[b_, w_] := 
  If[w > 1/2,
    If[b > 1/2,
      b,
      (2 w - 1) b + 1 - w
    ], (*else w<=1/2*)
    If[b > 1/2,
      - 2 w (1 - b) + 1,
      1 - w
    ]
  ]


HardNeuralAND[inputSize_, layerSize_] := InitializeBiasToZero[NetGraph[
  <|
    "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
    "WeightsClip" -> ElementwiseLayer[HardClip],
    "HardInclude" -> ThreadingLayer[HardAND[#Input, #Weights] &, 1, "Output" -> {layerSize, inputSize}],
    "Min" -> AggregationLayer[Min],
    "OutputClip" -> ElementwiseLayer[LogisticClip] 
  |>,
  {
    "Weights" -> "WeightsClip",
    "WeightsClip" -> NetPort["HardInclude", "Weights"],
    "HardInclude" -> "Min",
    "Min" -> "OutputClip"
  }
]]

HardNeuralNAND[inputSize_, layerSize_] := NetGraph[
  <|
    "AND" -> HardNeuralAND[inputSize, layerSize],
    "NOT" -> InitializeBiasToZero[NeuralNOT[layerSize]]
  |>,
  {
    "AND" -> "NOT"
  }
]

HardOR[b_, w_] := 1 - HardAND[1-b, w]

HardNeuralOR[inputSize_, layerSize_] := InitializeBiasToZero[NetGraph[
  <|
    "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
    "WeightsClip" -> ElementwiseLayer[HardClip],
    "HardInclude" -> ThreadingLayer[HardOR[#Input, #Weights] &, 1, "Output" -> {layerSize, inputSize}],
    "Max" -> AggregationLayer[Max],
    "OutputClip" -> ElementwiseLayer[LogisticClip]
  |>,
  {
    "Weights" -> "WeightsClip",
    "WeightsClip" -> NetPort["HardInclude", "Weights"],
    "HardInclude" -> "Max",
    "Max" -> "OutputClip"
  }
]]

HardNeuralNOR[inputSize_, layerSize_] := NetGraph[
  <|
    "OR" -> HardNeuralOR[inputSize, layerSize],
    "NOT" -> InitializeBalanced[NeuralNOT[layerSize]]
  |>,
  {
    "OR" -> "NOT"
  }
]

(* 
  Currently using sort (probably compiles to QuickSort)
    - average O(n log n)
    - worst-case O(n^2)
  Replace with Floyd-Rivest algorithm:
    - average n + \min(k, n - k) + O(\sqrt{n \log n}) comparisons with probability at least 1 - 2n^{-1/2}
    - worst-case O(n^2)
    - See https://danlark.org/2020/11/11/miniselect-practical-and-generic-selection-algorithms/
    - See also https://www.semanticscholar.org/paper/A-Fast-and-Flexible-Sorting-Algorithm-with-CUDA-Chen-Qin/648bd14cf19dd1a5ed5c09271bced0b3c6762dd9
      for sorting on GPUs.
*)
HardNeuralMajority[inputSize_, layerSize_] := With[
  {medianIndex = Ceiling[(inputSize + 1)/2]},
  InitializeBalanced[NetGraph[
    <|
      "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
      "WeightsClip" -> ElementwiseLayer[HardClip],
      "HardInclude" -> ThreadingLayer[HardNOT[#Input, #Weights] &, 1, "Output" -> {layerSize, inputSize}],
      "Sort" -> FunctionLayer[Sort /@ # &],
      "Medians" -> PartLayer[{All, medianIndex}],
      "OutputClip" -> ElementwiseLayer[HardClip]
    |>,
    {
      "Weights" -> "WeightsClip",
      "WeightsClip" -> NetPort["HardInclude", "Weights"],
      "HardInclude" -> "Sort",
      "Sort" -> "Medians",
      "Medians" -> "OutputClip"
    }
  ]]
]

PortLayer[inputSize_, numPorts_] := ReshapeLayer[{numPorts, inputSize / numPorts}]

AppendHardClassificationLoss[net_] := NetGraph[
  <|
    "NeuralLogicNet" -> net,
    "Probs" -> AggregationLayer[Mean],
    "loss" -> MeanSquaredLossLayer[]
  |>,  
  {
    "NeuralLogicNet" -> "Probs",
    "Probs" -> NetPort["loss", "Input"],
    "loss" -> NetPort["Loss"]
  }
]

(* ------------------------------------------------------------------ *)
(* Differentiable NOT, AND, OR *)
(* ------------------------------------------------------------------ *)

NeuralAND[inputSize_, layerSize_] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
    "WeightsClip" -> ElementwiseLayer[HardClip],
    "SoftInclude" -> ThreadingLayer[1 - #Weights (1 - #Input) &, 1, "Output" -> {layerSize, inputSize}],
    (* LogSumExp trick *)
    "And1" -> ElementwiseLayer[Log],
    "And2" -> AggregationLayer[Total],
    "And3" -> ElementwiseLayer[Exp],
    "OutputClip" -> ElementwiseLayer[HardClip] 
  |>,
  {
    "Weights" -> "WeightsClip",
    "WeightsClip" -> NetPort["SoftInclude", "Weights"],
    "SoftInclude" -> "And1",
    "And1" -> "And2",
    "And2" -> "And3",
    "And3" -> "OutputClip"
  }
]

NeuralOR[inputSize_, layerSize_] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Output" -> {layerSize, inputSize}],
    "WeightsClip" -> ElementwiseLayer[HardClip],
    "SoftInclude" -> ThreadingLayer[1 - #Weights #Input &, 1, "Output" -> {layerSize, inputSize}],
    (* LogSumExp trick *) 
    "Or1" -> ElementwiseLayer[Log],
    "Or2" -> AggregationLayer[Total],
    "Or3" -> ElementwiseLayer[Exp],
    "Or4" -> ElementwiseLayer[1 - # &],
    "OutputClip" -> ElementwiseLayer[HardClip]
  |>,
  {
    "Weights" -> "WeightsClip",
    "WeightsClip" -> NetPort["SoftInclude", "Weights"],
    "SoftInclude" -> "Or1",
    "Or1" -> "Or2",
    "Or2" -> "Or3",
    "Or3" -> "Or4",
    "Or4" -> "OutputClip"
  }
]

(* WIP *)

(*
BooleanMajorityLayer[input_, weights_] := Map[Inner[Xnor, input, #, Majority] &, weights]

BooleanMajorityChain[spec_List] := Function[{input, weights},
  Block[{activations = input},
    Scan[
      Block[{layerWeights = #},
        activations = BooleanMajorityLayer[activations, layerWeights];
      ] &,
      weights
    ];
    activations
  ]
] 
*)

(* ------------------------------------------------------------------ *)
(* Specify binary neural networks *)
(* ------------------------------------------------------------------ *)

(*
SetAttributes[BinaryNN, HoldFirst]

BinaryNN[spec_List] := Module[{softNet, hardNet},
  softNet = NetChain[# & /@ spec];
  (* TODO: update for mixed layers *)
  hardNet = BooleanMajorityChain[spec];
  {softNet, hardNet}
]
*)

(* ------------------------------------------------------------------ *)
(* Network hardening *)
(* ------------------------------------------------------------------ *)

(* TODO: update for mixed layers *)
(*
ExtractWeights[net_] := Module[{layers, layerSizes, majorityNames, majorityNeuronLayers},
  layers = NetExtract[net, All];
  layerSizes = Information[#, "ArraysCount"] & /@ layers;
  majorityNames = Map[Map[{"majority" <> ToString[#]} &, Range[#]] &, layerSizes];
  majorityNeuronLayers = MapThread[NetExtract, {layers, majorityNames}];
  Map[
    Map[Normal[NetExtract[NetExtract[#, "Weights"], "Arrays"]["Array"]] &, #] &,
    majorityNeuronLayers
  ]
]
*)

(* TODO: update for mixed layers *)
(*
HardBinaryNN[hardNet_, trainedSoftNet_] := Module[{hardenedWeights, inputSize},
  hardenedWeights = Harden[ExtractWeights[trainedSoftNet]];
  inputSize = Length[First[First[hardenedWeights]]];
  inputSymbols = Map[Symbol["x" <> ToString[#]] &, Range[inputSize]];
  Function[Evaluate[inputSymbols], Evaluate[hardNet[inputSymbols, hardenedWeights]]]
]
*)

End[]

EndPackage[]
