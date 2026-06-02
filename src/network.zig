const std = @import("std");

pub const Layer = struct {
    allocator: std.mem.Allocator,

    weights: []f32,
    biases:  []f32,
    inputs:    usize,
    outputs:   usize,

    last_input: []f32,
    last_output: []f32,

    // * m = average of gradients, v = average of squared gradients
    m_weights: []f32,
    v_weights: []f32,

    m_biases: []f32,
    v_biases: []f32,

    is_output: bool,

    t: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, inputs: usize, outputs: usize, is_output: bool) !Layer {
        var prng = std.Random.DefaultPrng.init(9445);
        const rand = prng.random();

        const weights = try allocator.alloc(f32, inputs * outputs);
        const biases = try allocator.alloc(f32, outputs);

        const m_weights = try allocator.alloc(f32, inputs * outputs);
        const v_weights = try allocator.alloc(f32, inputs * outputs);

        const m_biases = try allocator.alloc(f32, outputs);
        const v_biases = try allocator.alloc(f32, outputs);

        const last_input = try allocator.alloc(f32, inputs);
        const last_output = try allocator.alloc(f32, outputs);

        for (weights) |*v| {
            v.* = (rand.float(f32) * 2.0 - 1.0) * @sqrt(2.0 / @as(f32, @floatFromInt(inputs)));
        }

        for (biases) |*v| {
            v.* = 0;
        }

        for (m_weights) |*v| {
            v.* = 0;
        }

        for (v_weights) |*v| {
            v.* = 0;
        }

        for (m_biases) |*v| {
            v.* = 0;
        }

        for (v_biases) |*v| {
            v.* = 0;
        }

        return .{
            .allocator = allocator,
            .weights = weights,
            .biases = biases,
            .inputs = inputs,
            .outputs = outputs,
            .last_input = last_input,
            .last_output = last_output,
            .m_weights = m_weights,
            .v_weights = v_weights,
            .m_biases = m_biases,
            .v_biases = v_biases,
            .is_output = is_output,
        };
    }

    pub fn deinit(self: *Layer) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.biases);
        self.allocator.free(self.last_input);
        self.allocator.free(self.last_output);
        self.allocator.free(self.m_weights);
        self.allocator.free(self.v_weights);
        self.allocator.free(self.m_biases);
        self.allocator.free(self.v_biases);
    }

    pub fn forward(self: *Layer, input: []f32, output: []f32) void {
        for (output, 0..) |*output_v, i| {
            var sum: f32 = 0;
            for (input, 0..) |_, j| {
                sum += input[j] * self.weights[i * self.inputs + j];
            }

            const v = sum + self.biases[i];
            output_v.* = if (self.is_output) v else if (v > 0) v else 0.01 * v;
        }

        @memcpy(self.last_input, input);
        @memcpy(self.last_output, output);
    }

    pub fn backward(self: *Layer, output_gradient: []f32, input_gradient: []f32, learning_rate: f32) void {
        self.t += 1;

        for (0..self.outputs) |i| {
            const gate: f32 = if (self.is_output) 1.0 else if (self.last_output[i] > 0) 1.0 else 0.0;
            const delta: f32 = output_gradient[i] * gate;

            var m_b = self.m_biases[i];
            var v_b = self.v_biases[i];

            m_b = 0.9 * m_b + 0.1 * delta;
            v_b = 0.999 * v_b + 0.001 * (delta * delta);

            self.m_biases[i] = m_b;
            self.v_biases[i] = v_b;

            const m_b_hat = m_b / (1 - std.math.pow(f32, 0.9, @floatFromInt(self.t)));
            const v_b_hat = v_b / (1 - std.math.pow(f32, 0.999, @floatFromInt(self.t)));

            self.biases[i] -= learning_rate * m_b_hat / (@sqrt(v_b_hat) + 1e-8);

            for (0..self.inputs) |j| {
                input_gradient[j] += delta * self.weights[i * self.inputs + j];

                var m = self.m_weights[i * self.inputs + j];
                var v = self.v_weights[i * self.inputs + j];

                const grad = delta * self.last_input[j];
                m = 0.9 * m + 0.1 * grad;
                v = 0.999 * v + 0.001 * (grad * grad);

                self.m_weights[i * self.inputs + j] = m;
                self.v_weights[i * self.inputs + j] = v;

                const m_hat = m / (1 - std.math.pow(f32, 0.9, @floatFromInt(self.t)));
                const v_hat = v / (1 - std.math.pow(f32, 0.999, @floatFromInt(self.t)));

                self.weights[i * self.inputs + j] -= learning_rate * m_hat / (@sqrt(v_hat) + 1e-8);
            }
        }
    }
};

pub const Network = struct {
    allocator: std.mem.Allocator,
    buffers: [][]f32,
    layers: []Layer,
    max_layer_size: usize,

    pub fn init(allocator: std.mem.Allocator, layer_configs: []LayerConfig) !Network {
        var max_layer_size: usize = 0;
        for (layer_configs) |cfg| {
            if (cfg.inputs > max_layer_size) max_layer_size = cfg.inputs;
            if (cfg.outputs > max_layer_size) max_layer_size = cfg.outputs;
        }

        const layers = try allocator.alloc(Layer, layer_configs.len);

        for (layers, 0..) |*layer, idx| {
            layer.* = try Layer.init(allocator, layer_configs[idx].inputs,
                layer_configs[idx].outputs, layer_configs[idx].is_output);
        }

        const buffers = try allocator.alloc([]f32, layer_configs.len);
        for (buffers, 0..) |*buffer, idx| {
            buffer.* = try allocator.alloc(f32, layer_configs[idx].outputs);
        }

        return .{
            .allocator = allocator,
            .buffers = buffers,
            .layers = layers,
            .max_layer_size = max_layer_size,
        };
    }

    pub fn deinit(self: *Network) void {
        for (self.layers) |*layer| {
            layer.deinit();
        }

        for (self.buffers) |buffer| {
            self.allocator.free(buffer);
        }

        self.allocator.free(self.layers);
        self.allocator.free(self.buffers);
    }

    pub fn save(self: *Network) !void {
        const bin = try std.fs.cwd().createFile("network.znn", .{ .read = true });
        defer bin.close();

        const n_layers: u32 = @intCast(self.layers.len);
        try bin.writeAll(std.mem.asBytes(&n_layers));

        for (self.layers) |layer| {
            const inputs: u32 = @intCast(layer.inputs);
            const outputs: u32 = @intCast(layer.outputs);
            const is_out: u8 = if (layer.is_output) 1 else 0;
            try bin.writeAll(std.mem.asBytes(&inputs));
            try bin.writeAll(std.mem.asBytes(&outputs));
            try bin.writeAll(std.mem.asBytes(&is_out));
        }

        for (self.layers) |layer| {
            try bin.writeAll(std.mem.sliceAsBytes(layer.weights));
            try bin.writeAll(std.mem.sliceAsBytes(layer.biases));
        }
    }

    pub fn load(allocator: std.mem.Allocator, path: []const u8) !Network {
        const bin = try std.fs.cwd().openFile(path, .{});
        defer bin.close();

        var n_layers: u32 = 0;
        _ = try bin.readAll(std.mem.asBytes(&n_layers));

        const layer_configs = try allocator.alloc(LayerConfig, n_layers);
        for (layer_configs) |*cfg| {
            var inputs: u32 = 0;
            var outputs: u32 = 0;
            var is_out: u8 = 0;
            _ = try bin.readAll(std.mem.asBytes(&inputs));
            _ = try bin.readAll(std.mem.asBytes(&outputs));
            _ = try bin.readAll(std.mem.asBytes(&is_out));
            cfg.* = .{ .inputs = inputs, .outputs = outputs, .is_output = is_out == 1 };
        }

        const nwk = try Network.init(allocator, layer_configs);
        allocator.free(layer_configs);

        for (nwk.layers) |*layer| {
            _ = try bin.readAll(std.mem.sliceAsBytes(layer.weights));
            _ = try bin.readAll(std.mem.sliceAsBytes(layer.biases));
        }

        return nwk;
    }

    pub fn forward(self: *Network, input: []f32) []f32 {
        for (self.layers, 0..) |*layer, idx| {
            if (idx == 0) {
                layer.forward(input, self.buffers[idx]);
            } else {
                layer.forward(self.buffers[idx - 1], self.buffers[idx]);
            }
        }

        return self.buffers[self.layers.len - 1];
    }

    pub fn backward(self: *Network, expected_output: []f32, learning_rate: f32) !void {
        var gradient_a = try self.allocator.alloc(f32, self.max_layer_size);
        const gradient_b = try self.allocator.alloc(f32, self.max_layer_size);

        defer self.allocator.free(gradient_a);
        defer self.allocator.free(gradient_b);

        const act_output = self.buffers[self.buffers.len - 1];
        for (0..act_output.len) |i| {
            gradient_a[i] = act_output[i] - expected_output[i];
        }

        // std.debug.print("grad: {d:.4}\n", .{gradient_a[0]});

        var output_gradient = gradient_a;
        var input_gradient = gradient_b;

        var i = self.layers.len;
        while (i > 0) {
            i -= 1;

            @memset(input_gradient, 0);
            self.layers[i].backward(output_gradient[0..self.layers[i].outputs],
                input_gradient[0..self.layers[i].inputs], learning_rate);

            const temp_a = output_gradient;
            const temp_b = input_gradient;
            output_gradient = temp_b;
            input_gradient = temp_a;
        }
    }

    pub const LayerConfig = struct {
        inputs:  usize,
        outputs: usize,
        is_output: bool,
    };
};