NnEda is a personal C++ neural network library developed for dedicated usage in online programming competitions. Any backward-compatibility isn't guaranteed because code length is a critical factor in programming competitions and legacy functions get removed immediately. So, it's highly recommended to use this library as a reference for your own implementation.

Tested only on CodinGame.


## Usage of nn.cpp

The library exports trained networks as a string to be later embedded into source code. That's the only use case this library supports, and that's what the following code example does.

### Train and Export
```

// Only an MLP architecture is supported.
// Softmax is applied to the last layer.
// When a single probability score is needed, please let the last layer have 2 outputs.
// Then, the first output can be used as the probability.
vector<int> widths({4, 8, 8, 3});
nn_eda::Nn nn(widths);

// A momentum SGD optimizer is implemented..
for (Batch batch: data) {
  nn.train(batch.inputs, batch.labels, learning_rate);
}

// Models weights are encoded to a string using a Base64-wise encoding technique.
// The details are here: https://bowwowforeach.hatenablog.com/entry/2022/07/05/195417
nn_eda::NnIo::Obj io_obj = nn_eda::NnIo::to_obj(nn, encode_bits);
io_obj.write(out_path);

// A simple file IO is also supported.
nn_eda::NnIo::write_raw(nn, raw_out_path);
```


### Embed and Inference

```

// This is what the “write” method outputs.
// Copy-and-paste it from the file.
nn_eda::Nn nn = nn_eda::NnIo::from_obj(nn_eda::NnIo::Obj(
  std::vector<int>({4, 8, 8, 3, }),
  "㈪㊯㈠劧梩犹坜憐㨾嬆蓭䃡붌虷夗爷땝闂俻䃈縚朐㲚鵙䀖潈䶡鍗亶騀뇃祤늒査돥䯂礷뇖끈櫨념櫙㶼䉩㟀粃㮝睾㳝莺远缯佐蓛津䒔蕏괒僃洱䂩婞㚣壀꾻蟖䛖㤼甬喽凐뭢觎裬邛鱫餍矼杻㌀㘟鲉鞧漻吔漻吔漻吔漻吔漻吔漻吔漻吔漻刬",
  -0.66578274965286254883,
  0.72993195056915283203,
  10
));

// A simple file IO is also supported.
nn_eda::Nn nn = NnIo::read_raw(path);

// Perform inference.
// The “forward” method always returns the same float array.
// So, the returned values need to be used or saved before the next call.
float* output = nn.forward_softmax(input)

```

