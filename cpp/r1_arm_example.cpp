// Read the R1's joints, then optionally raise one by 30 degrees and lower it.
//
//   ./r1_arm_example eth0                  # READ ONLY: print the joint table
//   ./r1_arm_example eth0 --joint 22       # dry run: print the trajectory
//   ./r1_arm_example eth0 --joint 22 --go  # actually move
//
// Nothing moves without --go.
//
// SAFETY -- READ THIS
// -------------------
// The R1 exposes no rt/arm_sdk topic, so the only control path is rt/lowcmd,
// which is full low-level control of EVERY joint including the legs. Publish a
// frame with unset leg fields and a standing robot collapses.
//
// This program therefore holds every joint at the angle it was already at, and
// ramps only the one you name. Even so, it takes authority away from the
// robot's own balance controller, so:
//
//   * HANG OR SUPPORT THE ROBOT for the first run. Unitree's own H1 guide says
//     the same: "Running this routine will cause multiple joints to move. To
//     protect the robot, hang it before running this routine."
//   * put the robot in debug mode first, or lowcmd is ignored
//   * keep the e-stop in reach
//
// --joint is required and has no default. Published index tables are for the
// G1; the R1 has a different joint count and no public table, so guessing could
// drive a leg instead of the shoulder. Run without --joint to see the real
// layout, move the arm by hand, and note which index changes.

#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;

static const char *TOPIC_LOWCMD = "rt/lowcmd";
static const char *TOPIC_LOWSTATE = "rt/lowstate";

constexpr double PI = 3.14159265358979323846;
constexpr int CONTROL_HZ = 100;
constexpr double MAX_DEGREES = 60.0;  // hard cap regardless of input
constexpr float KP = 40.0f;           // low gains: compliant on contact
constexpr float KD = 1.0f;

static std::atomic<bool> g_have_state{false};
static LowState g_state;

// Unitree validates every LowCmd with this CRC. Without it the command is
// silently dropped -- the robot simply will not move and gives no error.
uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            } else {
                CRC32 <<= 1;
            }
            if (data & xbit) CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

void OnLowState(const void *message) {
    g_state = *(const LowState *)message;
    g_have_state = true;
}

double Deg(double rad) { return rad * 180.0 / PI; }
double Rad(double deg) { return deg * PI / 180.0; }

// Cosine easing: no acceleration step through the gearbox at either end.
double Ease(double phase) { return (1.0 - std::cos(PI * phase)) / 2.0; }

void PrintJoints() {
    const auto &motors = g_state.motor_state();
    std::cout << "\n" << motors.size() << " joints reporting\n\n";
    std::cout << "idx      q_rad     q_deg       dq     tau_est  temp\n";
    for (size_t i = 0; i < motors.size(); i++) {
        const auto &m = motors[i];
        if (m.q() == 0.0f && i > 30) continue;
        printf("%3zu  %9.4f  %8.2f  %7.3f  %8.3f  %d\n",
               i, m.q(), Deg(m.q()), m.dq(), m.tau_est(),
               (int)m.temperature()[0]);
    }
    std::cout << "\nMove the right arm by hand and re-run to see which index"
                 " changes.\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "usage: " << argv[0]
                  << " <network_interface> [--joint N] [--degrees D] [--go]\n"
                  << "  e.g. " << argv[0] << " eth0 --joint 22 --go\n";
        return 1;
    }

    std::string iface = argv[1];
    int joint = -1;
    double degrees = 30.0;
    bool go = false;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--joint" && i + 1 < argc) joint = std::stoi(argv[++i]);
        else if (a == "--degrees" && i + 1 < argc) degrees = std::stod(argv[++i]);
        else if (a == "--go") go = true;
    }

    if (std::fabs(degrees) > MAX_DEGREES) {
        std::cerr << "error: --degrees capped at " << MAX_DEGREES << "\n";
        return 1;
    }

    unitree::robot::ChannelFactory::Instance()->Init(0, iface);

    unitree::robot::ChannelSubscriberPtr<LowState> sub(
        new unitree::robot::ChannelSubscriber<LowState>(TOPIC_LOWSTATE));
    sub->InitChannel(OnLowState, 10);

    std::cout << "waiting for " << TOPIC_LOWSTATE << " on " << iface << " ...\n";
    for (int i = 0; i < 100 && !g_have_state; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!g_have_state) {
        std::cerr << "\nerror: no lowstate received.\n"
                  << "  * is the robot in DEBUG MODE?\n"
                  << "  * is this PC on 192.168.123.x, wired to the robot?\n"
                  << "  * is '" << iface << "' the right interface? (ifconfig)\n";
        return 1;
    }

    if (joint < 0) {
        PrintJoints();
        return 0;
    }

    const auto &motors = g_state.motor_state();
    if (joint >= (int)motors.size()) {
        std::cerr << "error: joint " << joint << " out of range (robot reports "
                  << motors.size() << ")\n";
        return 1;
    }

    const double start = motors[joint].q();
    const double target = start + Rad(degrees);
    const int travel_steps = 2 * CONTROL_HZ;   // 2s each way
    const int hold_steps = 1 * CONTROL_HZ;     // 1s at the top

    printf("\n  joint   : %d\n  start   : %+.2f deg\n  target  : %+.2f deg (%+.1f)\n",
           joint, Deg(start), Deg(target), degrees);
    printf("  profile : 2s up, 1s hold, 2s down at %d Hz\n", CONTROL_HZ);

    if (!go) {
        std::cout << "\n  DRY RUN -- nothing sent. Add --go to execute.\n";
        return 0;
    }

    std::cout << "\n  Robot should be HUNG OR SUPPORTED and in debug mode.\n"
              << "  Moving in 5s -- Ctrl-C aborts ...\n";
    for (int i = 5; i > 0; i--) {
        std::cout << "    " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    unitree::robot::ChannelPublisherPtr<LowCmd> pub(
        new unitree::robot::ChannelPublisher<LowCmd>(TOPIC_LOWCMD));
    pub->InitChannel();

    // Snapshot every joint so the rest of the body is commanded to stay exactly
    // where it is. Sending a frame with unset joints is what drops the robot.
    std::vector<float> hold(motors.size());
    for (size_t i = 0; i < motors.size(); i++) hold[i] = motors[i].q();

    LowCmd cmd{};
    auto send = [&](double q_target) {
        for (size_t i = 0; i < motors.size(); i++) {
            cmd.motor_cmd()[i].q() = hold[i];
            cmd.motor_cmd()[i].dq() = 0.0f;
            cmd.motor_cmd()[i].kp() = KP;
            cmd.motor_cmd()[i].kd() = KD;
            cmd.motor_cmd()[i].tau() = 0.0f;
        }
        cmd.motor_cmd()[joint].q() = (float)q_target;
        cmd.crc() = Crc32Core((uint32_t *)&cmd, (sizeof(LowCmd) >> 2) - 1);
        pub->Write(cmd);
    };

    const auto period = std::chrono::milliseconds(1000 / CONTROL_HZ);

    for (int i = 0; i < travel_steps; i++) {
        send(start + (target - start) * Ease((double)i / travel_steps));
        std::this_thread::sleep_for(period);
    }
    for (int i = 0; i < hold_steps; i++) {
        send(target);
        std::this_thread::sleep_for(period);
    }
    for (int i = 0; i < travel_steps; i++) {
        send(target + (start - target) * Ease((double)i / travel_steps));
        std::this_thread::sleep_for(period);
    }
    send(start);

    std::cout << "  done -- returned to start.\n";
    return 0;
}
