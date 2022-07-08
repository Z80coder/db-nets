(* ::Package:: *)

(* 
  TODO:
  - Support loss function on multiple bit output ports
  - bias term
*)

(* ------------------------------------------------------------------ *)
BeginPackage["neurallogic`"]
(* ------------------------------------------------------------------ *)

BinaryNN::usage = "Specifies a binary neural network and returns a {soft, hard} binary neural network pair.";
AppendLoss::usage = "Appends a loss function to a (soft) binary neural network.";
Harden::usage = "Convert soft-bit to hard-bit.";
Soften::usage = "Convert hard-bit to soft-bit.";
HardBinaryNN::usage = "Constructs a hard binary neural network from a trained soft binary neural network.";
ExtractWeights::usage = "Extract the weights from a soft neural network.";
NeuralMajority::usage = "Specifies a differentiable boolean majority function.";
WeightedNeuralMajority::usage = "Specifies a differentiable weighted boolean majority function.";
NeuralMajorityLayer::usage = "Specifies a neural majority layer.";
BooleanMajorityLayer::usage = "Specifies a boolean majority layer.";
BooleanMajorityChain::usage = "Specifies a boolean neural network chain.";
BitLoss::usage = "Specifies a bit loss function.";
NeuralMajorityForward::usage = "Forward pass of a neural majority layer.";
NeuralMajorityBackward::usage = "Backward pass of a neural majority layer.";
BitLossForward::usage = "Forward pass of a bit loss function.";
BitLossBackward::usage = "Backward pass of a bit loss function.";
RandomSoftBit::usage = "Randomly generate a soft bit.";
NeuralAND::usage = "Specifies a differentiable boolean AND function.";
NeuralANDLayer::usage = "Specifies a neural AND layer.";
NeuralOR::usage = "Specifies a differentiable boolean OR function.";
NeuralORLayer::usage = "Specifies a neural OR layer.";
NeuralNOT::usage = "Specifies a differentiable boolean NOT function.";
NeuralDNFLayer::usage = "Specifies a neural DNF layer.";
HardClip::usage = "Clips a hard bit between 0 and 1.";
LogisticClip::usage = "Clips a bit logistically between 0 and 1.";
HardIncludeAND::usage = "Differentiable hard AND logic.";
HardIncludeOR::usage = "Differentiable hard OR logic.";
HardNeuralAND::usage = "Differentiable hard AND logic.";
HardNeuralOR::usage = "Differentiable hard OR logic.";
Blip::usage = "Specifies a differentiable blip function.";
Stretch::usage = "Specifies a differentiable stretch function.";
InitializeNeuralLogicNet::usage = "Initialize a neural logic network.";
(*BinaryClassificationLayer::usage = "Specifies a binary classification layer.";*)

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
LogisticClip[x_] := LogisticSigmoid[5(2 x - 1)]

RandomSoftBit[n_] := RandomReal[{0.1, 0.9}, n]

(* ------------------------------------------------------------------ *)
(* Differentiable NOT, AND, OR *)
(* ------------------------------------------------------------------ *)

NeuralNOT[inputSize_] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Output" -> inputSize],
    "WeightsClip" -> ElementwiseLayer[HardClip], 
    "Not" -> ThreadingLayer[1 - #Weights + #Input (2 #Weights - 1) &, 1],
    "OutputClip" -> ElementwiseLayer[HardClip] 
  |>,
  {
    "Weights" -> "WeightsClip",
    "WeightsClip" -> NetPort["Not", "Weights"],
    "Not" -> "OutputClip"
  }
]

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

InitializeNeuralLogicNet[net_] := NetInitialize[
  net,
  Method -> {"Random", "Weights" -> CensoredDistribution[{0.001, 0.999}, NormalDistribution[-1, 1]]}
]

(* ------------------------------------------------------------------ *)
(* Differentiable HARD AND *)
(* ------------------------------------------------------------------ *)

(* TODO: compare hardening performance between these two options. *)

(* Simple, but doesn't always go through (1/2, 1/2). Bad*)
HardIncludeAND[b_, w_] := w b + 1 - w

(* Complex (piecewise linear) version of above that does always goo through (1/2, 1/2).*)
HardIncludeAND[b_, w_] := If[w > 1/2,
  If[b > 1/2,
    b,
    (2 w - 1) b + 1 - w
  ], (*else w<=1/2*)
  If[b > 1/2,
    - 2 w (1 - b) + 1,
    1 - w
  ]
]

(* Complex (piecewise linear) version of above that does always goo through (1/2, 1/2).*)
(* Existence of Blip seems to prevent learning ... unsure why *)
(*

Blip[x_, \[Epsilon]_] := 2 \[Epsilon] If[x > 1/2, 1 - x, x]

Stretch[x_] := 2 (x - 1/2)

HardIncludeAND[b_, w_] := Module[{r},
  r = If[w > 1/2,
    If[b > 1/2,
      b,
      (2 w - 1) b + 1 - w
    ], (*else w<=1/2*)
    If[b > 1/2,
      2 (Blip[Stretch[1 - w], 0.1] - w) (1 - b) + 1,
      2 Blip[Stretch[1 - w], 0.1] b + 1 - w
    ]
  ];
  r
]
*)

(*
HardIncludeAND[b_, w_] := If[w > 1/2, 
  1 - w + b (2 w - 1), 
  (* Not monotonic *)
  1 - w
  (*(w - 1/2) b + 1 - w*)
]
*)

(* Doesn't always go through (1/2, 1/2). Bad*)
(*
HardIncludeAND[b_, w_] := Module[{r},
  r = If[w > 1/2,
    If[b > 1/2,
      b,
      w b + 1/2(1 - w)
    ], (*else w<=1/2*)
    If[b > 1/2,
      (2 - (1/2)(3 - 2 w))b - 1 + 1/2(3 - 2w),
      (1/2)b + (1/2)(1 - w)
    ]
  ];
  r
] 
*)

(*
  Non-monotonic gradient. Bad
*)
(*
HardIncludeAND[b_, w_] := Module[{r},
  r = If[w > 1/2,
    (2 w - 1) b + 1 - w, 
     (*else w<=1/2*)
     (1/2 - w) b + 1/2
  ];
  r
]

HardIncludeAND[b_, w_] := If[w > 1/2,
  (1 - 2 (1 - w)) b + (1 - w),
  (* else w<=1/2 *)
  (1 - w - 1/2) b + 1/2
]
*)

(*
  Perhaps use this for faster min/max?
  https://en.wikipedia.org/wiki/Order_statistic_tree
  Or a hashing algorthm, but with log buckets clustered closer to 0.5.

  - Randomized algorithm with controllable error: start loose and then tighten?
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.4993&rep=rep1&type=pdf

  - https://en.algorithmica.org/hpc/algorithms/argmin/

*)

HardNeuralAND[weights_List] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Array" -> Drop[weights, -1]],
    "NonLin" -> ElementwiseLayer[ClipSoftBit],(*Ensure weights are [0,1]*)
    "SoftInclude" -> FunctionLayer[
      MapThread[HardIncludeAND[#1, #2] &, {#Input, #Weights}] &,
      "Input" -> Length[weights] - 1
    ],
    "And1" -> AggregationLayer[Min, All],
    (*
    "And1" -> FunctionLayer[
      Block[{mean = Mean[#Input], min = Min[#Input]}, 
        If[min > 1/2,
          (*min + (1 - mean) (min - 1/2),*)
          min,
          min + mean (1/2 - min)
        ]
      ]&,
      "Input" -> Length[weights] - 1
    ],
    *)
    "Reshape" -> ReshapeLayer[{1}],
    (*N.B. Output is not ensured to be in [0,1].This is taken care of by NeuralLogicLayer*)
    (* TODO: clip!! *)
    "Not" -> NeuralNOT[{Last[weights]}]
  |>,
  {
    "Weights" -> "NonLin",
    "NonLin" -> NetPort["SoftInclude", "Weights"],
    "SoftInclude" -> "And1",
    "And1" -> "Reshape",
    "Reshape" -> "Not"
  }
]


(*
HardNeuralAND[n_] := HardNeuralAND[RandomReal[{0.45, 0.55}, n + 1]]
NeuralANDLayer[inputSize_, layerSize_] := NeuralLogicLayer[inputSize, layerSize, HardNeuralAND[#] &]
*)

(* ------------------------------------------------------------------ *)
(* Differentiable HARD OR *)
(* ------------------------------------------------------------------ *)

HardIncludeOR[b_, w_] := 1 - HardIncludeAND[1-b, w]

(*
HardIncludeOR[b_, w_] := If[w > 1/2,
  (1 - 2 (1 - w)) b + (1 - w),
  (* else w<=1/2 *)
  (1/2 - w) b + w
]
*)

HardNeuralOR[weights_List] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Array" -> Drop[weights, -1]],
    "NonLin" -> ElementwiseLayer[ClipSoftBit], (* Ensure weights are [0,1]*)
    "SoftInclude" -> FunctionLayer[
      MapThread[HardIncludeOR[#1, #2] &, {#Input, #Weights}] &,
      "Input" -> Length[weights] - 1
    ],
    "Or1" -> AggregationLayer[Max, All],
    "Reshape" -> ReshapeLayer[{1}],
    "Not" -> NeuralNOT[{Last[weights]}]
    (*N.B. Output is not ensured to be in [0,1].This is taken care of by NeuralLogicLayer*)
  |>,
  {
    "Weights" -> "NonLin",
    "NonLin" -> NetPort["SoftInclude", "Weights"],
    "SoftInclude" -> "Or1",
    "Or1" -> "Reshape",
    "Reshape" -> "Not"
  }
]

(*
NeuralOR[n_] := NeuralOR[RandomReal[{0.1, 0.9}, n]]
NeuralORLayer[inputSize_, layerSize_] := NeuralLogicLayer[inputSize, layerSize, NeuralOR[#] &]
*)

(*
HardNeuralOR[n_] := HardNeuralOR[RandomReal[{0.45, 0.55}, n + 1]]
NeuralORLayer[inputSize_, layerSize_] := NeuralLogicLayer[inputSize, layerSize, HardNeuralOR[#] &]
*)

(* ------------------------------------------------------------------ *)
(* Differentiable DNF layer *)
(* ------------------------------------------------------------------ *)

(*
  TODO: work out the policy for the sizes that ensure all possible DNF expressions
  can be learned.
  TODO: then change parameter to [0, 1] from 0 capacity to full capacity to represent
  all possible DNF expressions.
  Ignoring NOTs then andSize does not need to be larger than 2^(inputSize - 1)
*)
NeuralDNFLayer[inputSize_, andSize_, orSize_] := NetGraph[
  <|
    (*"NOT layer 1" -> NeuralNOTLayer[inputSize],*)
    "AND layer" -> NeuralANDLayer[inputSize, andSize],
    "OR layer" -> NeuralORLayer[andSize, orSize]
  |>,
  {
    (*"NOT layer 1" -> "AND layer",*)
    "AND layer" -> "OR layer"
  }
]

(* ------------------------------------------------------------------ *)
(* Differentiable boolean majority *)
(* ------------------------------------------------------------------ *)

(*
  Computes the soft-majority of all input soft-bits.
  The input soft-bits are assumed to be between -1 and +1.
  The output is a soft-bit between -1 and +1.

  TODO.
*)
NeuralMajorityForward[] := Function[
  {
    Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]]
  },
  Block[
    {
      (* Currently using sort (probably compiles to QuickSort)
        - average O(n log n)
        - worst-case O(n^2)
      *)
      (* Replace with Floyd-Rivest algorithm:
        - average n + \min(k, n - k) + O(\sqrt{n \log n}) comparisons with probability at least 1 - 2n^{-1/2}
        - worst-case O(n^2)
        See https://danlark.org/2020/11/11/miniselect-practical-and-generic-selection-algorithms/
        - See also https://www.semanticscholar.org/paper/A-Fast-and-Flexible-Sorting-Algorithm-with-CUDA-Chen-Qin/648bd14cf19dd1a5ed5c09271bced0b3c6762dd9
        for sorting on GPUs.
        - What about training a small NN that takes values between 0 and 1 and then predicts arg min or arg max? Probably slower than just sorting.
      *)
      s = Sort[
        MapIndexed[
          {#1, #2[[1]]*1.0} &, 
          input
        ]
      ], 
      i = Floor[Ceiling[Length[input]/2.0]]
    },
    {First[s[[i]]], Last[s[[i]]]}
  ]
]
  
NeuralMajorityBackward[] := Function[
  {
    Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]],
    Typed[outgrad, TypeSpecifier["PackedArray"]["MachineReal", 1]],
    Typed[output, TypeSpecifier["PackedArray"]["MachineReal", 1]]
  },
  Block[
    {
      i = Round[Last[output]],
      g = First[outgrad]
    },
    Table[
      If[i == j, 
        g, (* Backprop the median bit *)
        If[i < j, 
          0.0, (* Make non-zero if we want to affect more than just median bit *)
          0.0
        ]
      ],
      {j, 1, Length[input]}
    ]
  ]
]

NeuralMajority[] := CompiledLayer[NeuralMajorityForward[], NeuralMajorityBackward[]]

(* ------------------------------------------------------------------ *)
(* Differentiable weighted boolean majority *)
(* ------------------------------------------------------------------ *)

WeightedNeuralMajority[weights_List] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Array" -> weights, "Output" -> Length[weights]],
    "WeightedBits" -> FunctionLayer[
      1 - #Bits + #Weights (2 #Bits - 1) &, 
      "Bits" -> Length[weights], 
      "Weights" -> Length[weights], 
      "Output" -> Length[weights]
    ],
    "NonLin" -> ElementwiseLayer[ClipSoftBit], 
    "Majority" -> NeuralMajority[],
    "MajorityBit" -> PartLayer[1],
    "Reshape" -> ReshapeLayer[{1}]
  |>,
  {
    "Weights" -> "WeightedBits",
    NetPort["Input"] -> "WeightedBits",
    "WeightedBits" -> "NonLin",
    "NonLin" -> "Majority",
    "Majority" -> "MajorityBit",
    "MajorityBit" -> "Reshape"
  }
]

WeightedNeuralMajority[n_] := WeightedNeuralMajority[RandomSoftBit[n]]

(* ------------------------------------------------------------------ *)
(* Majority layer *)
(* ------------------------------------------------------------------ *)

NeuralMajorityLayer[inputSize_, layerSize_] := NetGraph[
    Association[
      Map[
        "majority" <> ToString[#] -> WeightedNeuralMajority[inputSize] &, 
        Range[layerSize]
      ],
      "catenate" -> CatenateLayer[]
    ],
    Map[
      "majority" <> ToString[#] -> "catenate" &,
      Range[layerSize]
    ]
  ]

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

(* ------------------------------------------------------------------ *)
(* Specify binary neural networks *)
(* ------------------------------------------------------------------ *)

SetAttributes[BinaryNN, HoldFirst]

BinaryNN[spec_List] := Module[{softNet, hardNet},
  softNet = NetChain[# & /@ spec];
  (* TODO: update for mixed layers *)
  hardNet = BooleanMajorityChain[spec];
  {softNet, hardNet}
]

(* ------------------------------------------------------------------ *)
(* Bit loss *)
(* ------------------------------------------------------------------ *)

(* First half of input is the prediction, and the second half of the input is the target *)
BitLossForward[inputSize_, eps_] := Function[
  {Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]]},
  Block[
    {
      predictedBits = Take[input, inputSize], 
      targetBits = Take[input, -inputSize]
    },
    Table[
      (* TODO: unify with backward function *)
      If[(predictedBits[[n]] > (0.5 + eps) && targetBits[[n]] > 0.5) || (predictedBits[[n]] < (0.5 - eps) && targetBits[[n]] < 0.5),
        0.0,
        (* TODO: Perhaps measure error to hinge point? *)
        (predictedBits[[n]] - targetBits[[n]])^2
      ],
      {n, 1, inputSize}
    ]
  ]
]

BitLossBackward[inputSize_, eps_] := Function[
  {
    Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]],
    Typed[outgrad, TypeSpecifier["PackedArray"]["MachineReal", 1]]
  },
  Block[
    {
      predictedBits = Take[input, inputSize], 
      targetBits = Take[input, -inputSize]
    },
    Table[
      If[n <= inputSize,
        If[(predictedBits[[n]] > (0.5 + eps) && targetBits[[n]] > 0.5) || (predictedBits[[n]] < (0.5 - eps) && targetBits[[n]] < 0.5),
          0.0,
          2 (predictedBits[[n]] - targetBits[[n]])
        ],
        0.0
      ], 
      {n, 1, 2 inputSize}
    ]
  ]
]

(* 
  Small margin (e.g. 0.01) accelerates learning although trajectory is noisier. 
  Large margin (e.g. 0.5) slows learning but trajectory is more smooth.
*)
BitLoss[inputSize_] := With[{eps = 0.00001}, 
  CompiledLayer[
    BitLossForward[inputSize, eps], 
    BitLossBackward[inputSize, eps], 
    "Input" -> {2 inputSize}, 
    "Output" -> {inputSize}
  ]
]

(* ------------------------------------------------------------------ *)
(* Trainable (soft) binary neural network *)
(* ------------------------------------------------------------------ *)

(* 1 bit per class *)
AppendLoss[net_] := Block[{netOutputSize = NetExtract[net, "Output"]},
  NetGraph[
    <|
      "NeuralLogicNet" -> net,
      "Catenate" -> CatenateLayer["Input" -> {netOutputSize, netOutputSize}],
      "BitLoss" -> BitLoss[netOutputSize],
      "loss" -> SummationLayer[]
    |>,
    {
      "NeuralLogicNet" -> "Catenate",
      NetPort["Target"] -> "Catenate",
      "Catenate" -> "BitLoss",
      "BitLoss" -> "loss",
      "loss" -> NetPort["Loss"]
    }
  ]
]

(* n bits per class *)
(* Conceptually wrong*)
(*
BinaryClassificationLayer[net_, numClasses_] := Block[{
    netOutputSize = NetExtract[net, "Output"],
    outputPortSize
  },
  outputPortSize = netOutputSize / numClasses;
  NetGraph[
    <|
      "NeuralLogicNet" -> net,
      "Reshape" -> ReshapeLayer[{numClasses, outputPortSize}],
      "Probs" -> FunctionLayer[
        Map[
          (1.0 * Total[#]) / (1.0 * outputPortSize) &, 
          #Input
        ] &, 
        "Input" -> {numClasses, outputPortSize},
        "Output" -> {numClasses}
      ]
    |>,
    {
      "NeuralLogicNet" -> "Reshape",
      "Reshape" -> "Probs"
    }
  ]
]
*)

(* ------------------------------------------------------------------ *)
(* Network hardening *)
(* ------------------------------------------------------------------ *)

(* TODO: update for mixed layers *)
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

(* TODO: update for mixed layers *)
HardBinaryNN[hardNet_, trainedSoftNet_] := Module[{hardenedWeights, inputSize},
  hardenedWeights = Harden[ExtractWeights[trainedSoftNet]];
  inputSize = Length[First[First[hardenedWeights]]];
  inputSymbols = Map[Symbol["x" <> ToString[#]] &, Range[inputSize]];
  Function[Evaluate[inputSymbols], Evaluate[hardNet[inputSymbols, hardenedWeights]]]
]

End[]

EndPackage[]
