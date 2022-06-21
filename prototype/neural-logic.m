(* ::Package:: *)

BeginPackage["neurallogic`"]

BinaryNN::usage = "Specifies a binary neural network.";
Harden::usage = "Convert soft-bit to hard-bit.";

(* Added for testing only *)
NeuralMajority::usage = "Specifies a differentiable boolean majority function.";
WeightedNeuralMajority::usage = "Specifies a differentiable weighted boolean majority function.";
NeuralMajorityLayer::usage = "Specifies a neural majority layer.";
BitLoss::usage = "Specifies a bit loss function.";
NeuralMajorityForward::usage = "";
NeuralMajorityBackward::usage = "";
BitLossForward::usage = "";
BitLossBackward::usage = "";

Begin["`Private`"]

(* Boolean utilities *)

Harden[softBit_] := If[softBit > 0.5, 1, 0];

(* Differentiable boolean majority *)

NeuralMajorityForward[] := Function[
  {
    Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]]
  },
  Block[
    {
      (* TODO: replace Sort with more efficient median sort. *)
      s = Sort[MapIndexed[{#1, #2[[1]]*1.0} &, input]], 
      i = Ceiling[Length[input]/2.0]
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

(* Differentiable weighted boolean majority *)

WeightedNeuralMajority[weights_List] := NetGraph[
  <|
    "Weights" -> NetArrayLayer["Array" -> weights, "Output" -> Length[weights]],
    "WeightedBits" -> FunctionLayer[(1 - #Weights) + #Bits (2 #Weights - 1) &, 
    "Output" -> Length[weights]],
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

WeightedNeuralMajority[n_] := WeightedNeuralMajority[RandomReal[{0.45, 0.55}, n]]

(* Neural majority layer *)

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

(* Bit loss *)

(* First half of input is the prediction, and the second half of the input is the target *)
BitLossForward[inputSize_, eps_] := Function[
  {Typed[input, TypeSpecifier["PackedArray"]["MachineReal", 1]]},
  Block[
    {
      predictedBits = Take[input, inputSize], 
      targetBits = Take[input, -inputSize]
    },
    Table[
      If[(predictedBits[[n]] > (0.5 + eps) && targetBits[[n]] > 0.5) || (predictedBits[[n]] < (0.5 - eps) && targetBits[[n]] < 0.5),
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

BitLoss[inputSize_] := With[{eps = 0.45}, 
  CompiledLayer[
    BitLossForward[inputSize, eps], 
    BitLossBackward[inputSize, eps], 
    "Input" -> {2 inputSize}, 
    "Output" -> {inputSize}
  ]
]

(* Differentiable binary neural network *)

BinaryNN[net_] := Block[{netOutputSize = NetExtract[net, "Output"]},
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

End[]

EndPackage[]
