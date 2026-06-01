const std = @import("std");
const network = @import("network.zig");

const Network = network.Network;
const LayerConfig = Network.LayerConfig;

pub fn main() !void {

}

test "simple inputs" {
    const allocator = std.heap.page_allocator;

    var config = [2]LayerConfig{
        .{ .inputs = 2, .outputs = 3, .is_output = false },
        .{ .inputs = 3, .outputs = 1, .is_output = true },
    };

    var nwk: Network = try Network.init(allocator, &config);
    var input = [2]f32{ 2.0, 3.0 };

    const output = nwk.forward(&input);

    try std.testing.expectEqual(1, output.len);

    nwk.deinit();
}

test "network trains towards target" {
    const allocator = std.heap.page_allocator;

    var config = [2]LayerConfig{
        .{ .inputs = 2, .outputs = 4, .is_output = false },
        .{ .inputs = 4, .outputs = 1, .is_output = true },
    };

    var nwk: Network = try Network.init(allocator, &config);
    var input = [2]f32{ 0.5, 0.5 };
    var expected = [1]f32{ 1.0 };

    const before = nwk.forward(&input);
    std.debug.print("before: {d:.4}\n", .{before[0]});

    for (0..1000) |_| {
        _ = nwk.forward(&input);
        try nwk.backward(&expected, 0.01);
    }

    const after = nwk.forward(&input);
    std.debug.print("after: {d:.4}\n", .{after[0]});

    nwk.deinit();
}

test "network trains xor" {
    const allocator = std.heap.page_allocator;

    var config = [2]LayerConfig{
        .{ .inputs = 2, .outputs = 64, .is_output = false },
        .{ .inputs = 64, .outputs = 1, .is_output = true },
    };

    var nwk: Network = try Network.init(allocator, &config);

    var input2 = [2]f32{ 0.0, 1.0 };
    var expected2 = [1]f32{ 1.0 };

    var input3 = [2]f32{ 1.0, 0.0 };
    var expected3 = [1]f32{ 1.0 };

    var input4 = [2]f32{ 1.0, 1.0 };
    var expected4 = [1]f32{ 0.0 };

    for (0..100000) |_| {
        _ = nwk.forward(&input2); try nwk.backward(&expected2, 0.01);
        _ = nwk.forward(&input3); try nwk.backward(&expected3, 0.01);
        _ = nwk.forward(&input4); try nwk.backward(&expected4, 0.01);
    }

    const r2 = nwk.forward(&input2)[0];
    std.debug.print("r2: {d:.4}\n", .{r2});
    const r3 = nwk.forward(&input3)[0];
    std.debug.print("r3: {d:.4}\n", .{r3});
    const r4 = nwk.forward(&input4)[0];
    std.debug.print("r4: {d:.4}\n", .{r4});

    nwk.deinit();
}