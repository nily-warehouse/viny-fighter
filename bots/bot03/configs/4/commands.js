const TURN_STEP = Math.PI / 12;

const COMMANDS = {
    "noop":
        {},
    "turn_left":
        { "dyaw": TURN_STEP },
    "turn_right":
        { "dyaw": -TURN_STEP },
    "attack":
        { "attack": true },
};

module.exports = { COMMANDS, TURN_STEP };