const path = require("path");
const dotenv = require("dotenv");
const mineflayer = require("mineflayer");
const { MinecraftEnv, Transition, Decision } = require("./env");
const { TransitionWriter } = require("./recorder");


// --- Configs ---

// - env configs -

const env_path = path.resolve(__dirname, "..", ".env");
dotenv.config({ path: env_path });

let   NAME = process.env.BOTNAME;
const RANGE = Number.parseInt(process.env.RANGE, 10);
const HOST = process.env.HOST;
const PORT = 25565
const VERSION = process.env.MINECRAFT_VERSION;
const EPISODE_TIMEOUT_SECONDS = Number(process.env.EPISODE_TIMEOUT_SECONDS);

if (!Number.isFinite(EPISODE_TIMEOUT_SECONDS) || EPISODE_TIMEOUT_SECONDS <= 0) {
    throw new Error("EPISODE_TIMEOUT_SECONDS must be a positive number");
}

const EPISODE_TIMEOUT_MS = EPISODE_TIMEOUT_SECONDS * 1000;

let teleporting = false;
let tp_ticks = 0;
const TP_TIMEOUT_TICKS = 40;


// - initial setup -

const args = process.argv.slice(2);

let tick_counter = 0;
let running = false;
let slot = 0;
let policy = null;
let epLimit = -1;
let config = 1;
let episode_timer = null;
let pending_attack = null;
const attacked_targets = new Map();
const combat_events = [];

for (const arg of args) {
  if (arg === "instant-run") {
    running = true;
    console.log('[INFO]: instant-run activated')
    continue;
  }

  const separatorIndex = arg.indexOf(":");

  if (separatorIndex === -1) {
    continue;
  }

  const key = arg.slice(0, separatorIndex);
  const value = arg.slice(separatorIndex + 1);

  if (key === "slot") {
    const parsedSlot = Number(value);

    if (!/^\d+$/.test(value) || !Number.isSafeInteger(parsedSlot)) {
      console.error("Enter a positive natural number!");
      process.exit(1);
    }

    slot = parsedSlot;
    NAME += slot
  }
  
  else if (key === "policy") {
    if (value.trim() === "") {
      console.error("Enter a positive natural number!");
      process.exit(1);
    }

    policy = value;
  }

  else if (key === "config") {
    const parsedConfig = Number(value);

    if (!/^\d+$/.test(value) || !Number.isSafeInteger(parsedConfig) || parsedConfig < 1) {
      console.error("Config must be a positive integer!");
      process.exit(1);
    }

    config = parsedConfig;
  }

  else if (key === "ep") {
    const parsedEP = Number(value)
    
    if (!/^\d+$/.test(value) || !Number.isSafeInteger(parsedEP)) {
      console.error("Enter a positive natural number!");
      process.exit(1);
    }

    epLimit = value;
  }
}

const environment = new MinecraftEnv(policy, epLimit, 2, config);
const writer = new TransitionWriter(slot);


// --- Essentials Connection things ---

const bot = mineflayer.createBot({
    username: NAME,
    host: HOST,
    port: PORT,
    version: VERSION,
    hideErrors: true
});

function login() {
    const bot_socket = bot._client.socket;
    console.log(
        `[INFO]: Logged in to ${bot_socket.server || bot_socket._host}`
    );
}

function kicked(reason, loggedIn) {
    if (loggedIn) {
        console.log(`[INFO]: Kicked from server: ${reason}`);
    } else {
        console.log(`[INFO]: Kicked whilst trying to connect: ${reason}`);
    }
}

function end(reason) {
    pending_attack = null;
    clear_episode_timer();
    console.log(`[INFO]: Disconnected: ${reason}`);

    bot.removeListener("login", login);
    bot.removeListener("kicked", kicked);
    bot.removeListener("end", end);
}

function setSlot() {
    if (slot > 0) {
        bot.chat('/mv tp slot_' + slot);
    }
}

function spawn() {
    killed_ids.clear()
    attacked_targets.clear();
    combat_events.length = 0;

    pending_attack = null;
    clear_episode_timer();

    if (environment.limit_reached) {
        running = false;
        console.log("Episode Limit reached");
        bot.quit();
        return
    }

    teleporting = true;
    send_tp();
}

function near_spawn() {
    const p = bot.entity.position;
    return Math.abs(p.x - 8.5) < 0.5 && Math.abs(p.y + 60) < 0.5 && Math.abs(p.z - 8.5) < 0.5;
}

function send_tp() {
    bot.clearControlStates();
    bot.chat(`/tp ${NAME} 8.5 -60 8.5`);
    tp_ticks = 0;
}

function clear_episode_timer() {
    if (episode_timer !== null) {
        clearTimeout(episode_timer);
        episode_timer = null;
    }
}

function start_episode_timer() {
    clear_episode_timer();
    episode_timer = setTimeout(() => {
        episode_timer = null;
        console.log(
            `[INFO]: Episode ${environment.episode} reached the ` +
            `${EPISODE_TIMEOUT_SECONDS}s timeout`
        );
        bot.chat(`/kill ${NAME}`);
    }, EPISODE_TIMEOUT_MS);
}

bot.on("login", login);
bot.once('spawn', setSlot);
bot.on('spawn', spawn);
bot.on("kicked", kicked);
bot.on("end", end);

// --- Running and Death Managment ---

bot.on("death", () => {
    clear_episode_timer();
    console.log(`[INFO]: Episode ${environment.episode} just finished!`);

    environment.on_death(get_state());
    pending_attack = null;
    write_transition(environment.pop_transitions());
});

bot.on("messagestr", (message, messagePosition) => {
    if (messagePosition !== "chat") {
        return;
    }

    const prefix = `bot ${NAME}`;
    const parts = message.split(" ");

    if (parts.length > 3 && parts[1] === "bot" && parts[2] === NAME) {
        const command = message.split(prefix, 2)[1].trim();

        if (command === "quit") {
            environment.on_death(get_state());
            write_transition(environment.pop_transitions());
            bot.quit();
        } else if (command === "run") {
            running = true;
            console.log("[INFO]: Turned on")
        } else if (command === "stop") {
            running = false;
            pending_attack = null;
            bot.clearControlStates();
            console.log("[INFO]: Turned off")
        } else {
            console.log(`${parts[0].slice(1, -1)} just said: ${command}`);
        }
    }
});


// --- Every tick trigger ---

function execute() {
    if (running) {
        const action = environment.on_tick(get_state());
        perform_action(action);
    }
}

bot.on("physicsTick", () => {
    if (teleporting) {
        if (near_spawn()) {
            teleporting = false;
            start_episode_timer();
        } else if (++tp_ticks >= TP_TIMEOUT_TICKS) {
            send_tp();
        }
        return;
    }

    tick_counter += 1;

    if (tick_counter >= environment.tick_every) {
        tick_counter = 0;
        execute();
    }
});


// --- Action ---

const MAX_PITCH = Math.PI / 2 - 0.01;

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function normalize_yaw(yaw) {
    return ((yaw + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

function perform_action(action) {
    pending_attack = null;
    bot.clearControlStates();

    if ("controls" in action) {
        for (const control of action.controls) {
            bot.setControlState(control, true);
        }
    }

    if ("dyaw" in action || "dpitch" in action) {
        const new_yaw = normalize_yaw(bot.entity.yaw + (action.dyaw || 0));
        const new_pitch = clamp(
            bot.entity.pitch + (action.dpitch || 0),
            -MAX_PITCH,
            MAX_PITCH
        );
        bot.look(new_yaw, new_pitch);
    }

    if ("attack" in action) {
        const target = bot.entityAtCursor(RANGE);
        if (target) {
            const health = get_zombie_health(target);
            if (health !== null && health > 0) {
                pending_attack = { target, health };
                if (!attacked_targets.has(target.id)) {
                    attacked_targets.set(target.id, { health, killed: false });
                }
            }
            bot.attack(target);
        } else {
            bot.swingArm("right");
        }
    }
}


// --- Game State Interpretation ---

function get_zombie_health(entity) {
    if (!entity || entity.name !== "zombie") {
        return null;
    }

    const metadata_keys = bot.registry.entitiesByName.zombie?.metadataKeys;
    const health_index = metadata_keys ? metadata_keys.indexOf("health") : -1;
    const health = health_index >= 0 ? entity.metadata?.[health_index] : null;
    return Number.isFinite(health) && health >= 0 ? health : null;
}

function get_state() {
    const entity = bot.entity;
    const target = get_nearest_zombie();
    const offset = target ? matrixAdition(target.position, entity.position) : null;
    const attacked_health = pending_attack ? get_zombie_health(pending_attack.target) : null;

    return {
        // bot state
        position: vector_to_tuple(entity.position),
        velocity: vector_to_tuple(entity.velocity),
        yaw: entity.yaw,
        pitch: entity.pitch,
        on_ground: entity.onGround,
        health: bot.health,

        // target
        target: offset,
        target_id: target ? target.id : null,
        target_position: target ? vector_to_tuple(target.position) : null,
        target_health: get_zombie_health(target),
        attack_hit: pending_attack !== null && attacked_health !== null &&
            attacked_health < pending_attack.health,
        target_killed: pending_attack !== null && killed_ids.has(pending_attack.target.id),
        combat_events: combat_events.splice(0),
    };
}

function vector_to_tuple(vector) {
    return [vector.x, vector.y, vector.z];
}

const killed_ids = new Set();
bot.on("entityUpdate", entity => record_combat_update(entity));
bot.on("entityDead", entity => {
    if (entity.name === "zombie") {
        killed_ids.add(entity.id);
        record_combat_update(entity, true);
    }
});

function record_combat_update(entity, died = false) {
    const tracked = attacked_targets.get(entity.id);
    if (!tracked || tracked.killed) return;
    const health = died ? 0 : get_zombie_health(entity);
    if (health === null) return;
    const damage = Math.max(0, tracked.health - health);
    const killed = died || health <= 0;
    tracked.health = health;
    tracked.killed = killed;
    if (damage > 0 || killed) {
        combat_events.push({ target_id: entity.id, damage, killed });
    }
}

function get_nearest_zombie() {
    return bot.nearestEntity(e => {
        if (e.name !== "zombie") return false;
        const h = get_zombie_health(e);
        return h === null || h > 0;
    });
}


// --- Write transition ---

function write_transition(transitions) {
    writer.append(transitions);
}


// --- Utils ---

function matrixAdition(a, b) {
    return [a.x - b.x, a.y - b.y, a.z - b.z]
}


module.exports = { bot, environment, get_state, perform_action };
