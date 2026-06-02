const std = @import("std");
const network = @import("network.zig");

const Network = network.Network;
const LayerConfig = Network.LayerConfig;

pub fn main() !void {

}

test "sine save and load" {
    const allocator = std.heap.page_allocator;

    // var config = [4]LayerConfig{
    //     .{ .inputs = 1, .outputs = 64, .is_output = false },
    //     .{ .inputs = 64, .outputs = 32, .is_output = false },
    //     .{ .inputs = 32, .outputs = 16, .is_output = false },
    //     .{ .inputs = 16, .outputs = 1, .is_output = true },
    // };
    //
    // var nwk = try Network.init(allocator, &config);
    // defer nwk.deinit();
    //
    // var prng = std.Random.DefaultPrng.init(42);
    // const rand = prng.random();
    //
    // for (0..500000) |_| {
    //     const x = rand.float(f32);
    //     const expected = (std.math.sin(x * std.math.pi * 2.0) + 1.0) / 2.0;
    //     var input = [1]f32{ x };
    //     var exp = [1]f32{ expected };
    //     _ = nwk.forward(&input);
    //     try nwk.backward(&exp, 0.001);
    // }
    //
    // try nwk.save();
    // std.debug.print("saved\n", .{});
    //
    var loaded = try Network.load(allocator, "network.znn");
    defer loaded.deinit();

    const test_points = [_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 };
    for (test_points) |x| {
        var input = [1]f32{ x };
        const result = loaded.forward(&input)[0];
        const expected = (std.math.sin(x * std.math.pi * 2.0) + 1.0) / 2.0;
        std.debug.print("sin({d:.2}pi) = {d:.4} (expected {d:.4})\n", .{ x * 2.0, result, expected });
    }
}

// test "save and load" {
//     const allocator = std.heap.page_allocator;
//
//     var config = [2]LayerConfig{
//         .{ .inputs = 2, .outputs = 8, .is_output = false },
//         .{ .inputs = 8, .outputs = 1, .is_output = true },
//     };
//
//     var nwk = try Network.init(allocator, &config);
//     defer nwk.deinit();
//
//     var input = [2]f32{ 0.5, 0.5 };
//     var expected = [1]f32{ 1.0 };
//
//     for (0..1000) |_| {
//         _ = nwk.forward(&input);
//         try nwk.backward(&expected, 0.01);
//     }
//
//     const before = nwk.forward(&input)[0];
//     try nwk.save();
//
//     var loaded = try Network.load(allocator, "network.znn");
//     defer loaded.deinit();
//
//     const after = loaded.forward(&input)[0];
//
//     std.debug.print("before save: {d:.6}\n", .{before});
//     std.debug.print("after load:  {d:.6}\n", .{after});
//
//     try std.testing.expectApproxEqAbs(before, after, 0.0001);
// }

// test "simple inputs" {
//     const allocator = std.heap.page_allocator;
//
//     var config = [2]LayerConfig{
//         .{ .inputs = 2, .outputs = 3, .is_output = false },
//         .{ .inputs = 3, .outputs = 1, .is_output = true },
//     };
//
//     var nwk: Network = try Network.init(allocator, &config);
//     var input = [2]f32{ 2.0, 3.0 };
//
//     const output = nwk.forward(&input);
//
//     try std.testing.expectEqual(1, output.len);
//
//     nwk.deinit();
// }
//
// test "multiplication generalisation" {
//     const allocator = std.heap.page_allocator;
//
//     var config = [3]LayerConfig{
//         .{ .inputs = 2, .outputs = 64, .is_output = false },
//         .{ .inputs = 64, .outputs = 32, .is_output = true },
//         .{ .inputs = 32, .outputs = 1, .is_output = true },
//     };
//
//     var nwk: Network = try Network.init(allocator, &config);
//
//     var prng = std.Random.DefaultPrng.init(12345);
//     const rand = prng.random();
//
//     for (0..100000) |_| {
//         const a = rand.float(f32) * 10.0;
//         const b = rand.float(f32) * 10.0;
//         var input = [2]f32{ a / 10.0, b / 10.0 };
//         var expected = [1]f32{ (a * b) / 100.0 };
//         _ = nwk.forward(&input);
//         try nwk.backward(&expected, 0.001);
//     }
//
//     const pairs = [_][2]f32{ .{3.0, 7.0}, .{9.0, 2.0}, .{5.0, 5.0}, .{8.0, 4.0} };
//     for (pairs) |pair| {
//         var input = [2]f32{ pair[0] / 10.0, pair[1] / 10.0 };
//         const result = nwk.forward(&input)[0] * 100.0;
//         std.debug.print("{d:.1} * {d:.1} = {d:.4} (expected {d:.4})\n", .{ pair[0], pair[1], result, pair[0] * pair[1] });
//     }
//
//     nwk.deinit();
// }
//
// test "sine approximation" {
//     const allocator = std.heap.page_allocator;
//
//     var config = [4]LayerConfig{
//         .{ .inputs = 1, .outputs = 64, .is_output = false },
//         .{ .inputs = 64, .outputs = 32, .is_output = false },
//         .{ .inputs = 32, .outputs = 16, .is_output = false },
//         .{ .inputs = 16, .outputs = 1, .is_output = true },
//     };
//
//     var nwk = try Network.init(allocator, &config);
//     defer nwk.deinit();
//
//     var prng = std.Random.DefaultPrng.init(42);
//     const rand = prng.random();
//
//     for (0..500000) |_| {
//         const x = rand.float(f32);
//         const expected = (std.math.sin(x * std.math.pi * 2.0) + 1.0) / 2.0;
//         var input = [1]f32{ x };
//         var exp = [1]f32{ expected };
//         _ = nwk.forward(&input);
//         try nwk.backward(&exp, 0.001);
//     }
//
//     const test_points = [_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 };
//     for (test_points) |x| {
//         var input = [1]f32{ x };
//         const result = nwk.forward(&input)[0];
//         const expected = (std.math.sin(x * std.math.pi * 2.0) + 1.0) / 2.0;
//         std.debug.print("sin({d:.2}π) = {d:.4} (expected {d:.4})\n", .{ x * 2.0, result, expected });
//     }
// }
//
// test "network trains towards target" {
//     const allocator = std.heap.page_allocator;
//
//     var config = [2]LayerConfig{
//         .{ .inputs = 2, .outputs = 4, .is_output = false },
//         .{ .inputs = 4, .outputs = 1, .is_output = true },
//     };
//
//     var nwk: Network = try Network.init(allocator, &config);
//     var input = [2]f32{ 0.5, 0.5 };
//     var expected = [1]f32{ 1.0 };
//
//     const before = nwk.forward(&input);
//     std.debug.print("before: {d:.4}\n", .{before[0]});
//
//     for (0..1000) |_| {
//         _ = nwk.forward(&input);
//         try nwk.backward(&expected, 0.01);
//     }
//
//     const after = nwk.forward(&input);
//     std.debug.print("after: {d:.4}\n", .{after[0]});
//
//     nwk.deinit();
// }
//
// test "network trains xor" {
//     const allocator = std.heap.page_allocator;
//
//     var config = [2]LayerConfig{
//         .{ .inputs = 2, .outputs = 8, .is_output = false },
//         .{ .inputs = 8, .outputs = 1, .is_output = true },
//     };
//
//     var nwk: Network = try Network.init(allocator, &config);
//
//     var input2 = [2]f32{ 0.0, 1.0 };
//     var expected2 = [1]f32{ 1.0 };
//
//     var input3 = [2]f32{ 1.0, 0.0 };
//     var expected3 = [1]f32{ 1.0 };
//
//     var input4 = [2]f32{ 1.0, 1.0 };
//     var expected4 = [1]f32{ 0.0 };
//
//     for (0..100000) |_| {
//         _ = nwk.forward(&input2); try nwk.backward(&expected2, 0.01);
//         _ = nwk.forward(&input3); try nwk.backward(&expected3, 0.01);
//         _ = nwk.forward(&input4); try nwk.backward(&expected4, 0.01);
//     }
//
//     const r2 = nwk.forward(&input2)[0];
//     std.debug.print("r2: {d:.4}\n", .{r2});
//     const r3 = nwk.forward(&input3)[0];
//     std.debug.print("r3: {d:.4}\n", .{r3});
//     const r4 = nwk.forward(&input4)[0];
//     std.debug.print("r4: {d:.4}\n", .{r4});
//
//     nwk.deinit();
// }