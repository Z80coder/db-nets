(* ::Package:: *)

(* 
  TODO:
  - Harden the entire net to a boolean function
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

(* Added for testing only *)
NeuralMajority::usage = "Specifies a differentiable boolean majority function.";
WeightedNeuralMajority::usage = "Specifies a differentiable weighted boolean majority function.";
NeuralMajorityLayer::usage = "Specifies a neural majority layer.";
BooleanMajorityLayer::usage = "Specifies a boolean majority layer.";
BooleanNetChain::usage = "Specifies a boolean neural network chain.";
BitLoss::usage = "Specifies a bit loss function.";
NeuralMajorityForward::usage = "Forward pass of a neural majority layer.";
NeuralMajorityBackward::usage = "Backward pass of a neural majority layer.";
BitLossForward::usage = "Forward pass of a bit loss function.";
BitLossBackward::usage = "Backward pass of a bit loss function.";

(* ------------------------------------------------------------------ *)
Begin["`Private`"]
(* ------------------------------------------------------------------ *)

(* ------------------------------------------------------------------ *)
(* Boolean utilities *)
(* ------------------------------------------------------------------ *)

(* Soft-bits between -1 and +1 *)
Harden[softBit_] := If[softBit > 0.0, True, False]
Harden[softBit_List] := Map[Harden, softBit]

(* Hard-bits between 0 and 1 *)
Soften[hardBit_] := If[hardBit == 1, 1.0, -1.0]

(* ------------------------------------------------------------------ *)
(* Differentiable boolean majority *)
(* ------------------------------------------------------------------ *)

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
      *)
      s = Sort[MapIndexed[{#1, #2[[1]]*1.0} &, input]], 
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
      If[i == j, g, If[i < j, 0.0, 0.0]],
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
    "WeightedBits" -> FunctionLayer[Clip[#Weights #Bits, {-1, 1}] &, "Output" -> Length[weights]],
    "Majority" -> NeuralMajority[],
    "MajorityBit" -> PartLayer[1],
    "Reshape" -> ReshapeLayer[{1}]
  |>,
  {
    "Weights" -> "WeightedBits",
    NetPort["Input"] -> "WeightedBits",
    "WeightedBits" -> "Majority",
    "Majority" -> "MajorityBit",
    "MajorityBit" -> "Reshape"
  }
]

WeightedNeuralMajority[n_] := WeightedNeuralMajority[RandomReal[{-0.2, 0.2}, n]]

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

BooleanNetChain[spec_List] := Function[{input, weights},
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

BinaryNN[spec_List] := Module[{softNet, hardNet},
  softNet = NetChain[NeuralMajorityLayer @@ # & /@ spec];
  hardNet = BooleanNetChain[spec];
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
      If[(predictedBits[[n]] > (0.0 + eps) && targetBits[[n]] > 0.0) || (predictedBits[[n]] < (0.0 - eps) && targetBits[[n]] < 0.0),
        0.0,
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
        If[(predictedBits[[n]] > (0.0 + eps) && targetBits[[n]] > 0.0) || (predictedBits[[n]] < (0.0 - eps) && targetBits[[n]] < 0.0),
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
  Large margin (e.g. 1.0) slows learning but trajectory is more smooth.
*)
BitLoss[inputSize_] := With[{eps = 1.0}, 
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

AppendLoss[net_] := Block[{netOutputSize = NetExtract[net, "Output"]},
  NetGraph[
    <|
      "MajorityNet" -> net,
      "Catenate" -> CatenateLayer[],
      "BitLoss" -> BitLoss[netOutputSize],
      "loss" -> SummationLayer[]
    |>,
    {
      "MajorityNet" -> "Catenate",
      NetPort["Target"] -> "Catenate",
      "Catenate" -> "BitLoss",
      "BitLoss" -> "loss",
      "loss" -> NetPort["Loss"]
    }
  ]
]

(* ------------------------------------------------------------------ *)
(* Network hardening *)
(* ------------------------------------------------------------------ *)

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

HardBinaryNN[hardNet_, trainedSoftNet_] := Module[{hardenedWeights, inputSize},
  hardenedWeights = Harden[ExtractWeights[trainedSoftNet]];
  inputSize = Length[First[First[hardenedWeights]]];
  inputSymbols = Map[Symbol["x" <> ToString[#]] &, Range[inputSize]];
  Function[Evaluate[inputSymbols], Evaluate[hardNet[inputSymbols, hardenedWeights]]]
]

End[]

EndPackage[]
