{ config, pkgs, ... }:
{
  imports = [ ./hardware-configuration.nix ];

  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  boot.kernelPackages = pkgs.linuxPackages_latest;
  boot.kernelParams = [ "acpi_enforce_resources=lax" ];

  networking.hostName = "captain";
  networking.networkmanager.enable = true;
  networking.firewall.enable = false;

  time.timeZone = "Europe/Oslo";
  i18n.defaultLocale = "en_US.UTF-8";

  services.xserver.enable = true;
  services.displayManager.gdm.enable = true;
  services.desktopManager.gnome.enable = true;
  services.xserver.xkb = { layout = "no"; variant = "nodeadkeys"; };
  console.keyMap = "no";

  services.printing.enable = true;
  services.pulseaudio.enable = false;
  security.rtkit.enable = true;
  services.pipewire = {
    enable = true;
    alsa.enable = true;
    alsa.support32Bit = true;
    pulse.enable = true;
  };


  # Disable sleep/suspend/hibernate completely
  systemd.targets.sleep.enable = false;
  systemd.targets.suspend.enable = false;
  systemd.targets.hibernate.enable = false;
  systemd.targets.hybrid-sleep.enable = false;
  services.logind.lidSwitch = "ignore";
  services.logind.lidSwitchExternalPower = "ignore";
  services.logind.settings.Login.IdleAction = "ignore";
  powerManagement.enable = false;

  services.openssh.enable = true;
  services.openssh.settings.PermitRootLogin = "yes";
  services.openssh.settings.PasswordAuthentication = true;

  # NVIDIA
  hardware.graphics.enable = true;
  services.xserver.videoDrivers = [ "nvidia" ];
  hardware.nvidia = {
    modesetting.enable = true;
    open = true;
    nvidiaSettings = true;
    package = config.boot.kernelPackages.nvidiaPackages.stable;
  };

  # OpenRGB - disable all RGB lights
  services.hardware.openrgb.enable = true;
  services.hardware.openrgb.motherboard = "amd";

  systemd.services.rgb-off = {
    description = "Turn off all RGB lights";
    after = [ "openrgb.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "oneshot";
      ExecStart = let
        script = pkgs.writeShellScript "rgb-off" ''
          sleep 3
          ${pkgs.openrgb}/bin/openrgb --noautoconnect -d 0 -m direct -c 000000
          ${pkgs.openrgb}/bin/openrgb --noautoconnect -d 1 -m custom -c 000000
        '';
      in "${script}";
      RemainAfterExit = true;
    };
  };

  # User
  users.users.friend = {
    isNormalUser = true;
    description = "friend";
    extraGroups = [ "networkmanager" "wheel" "video" "podman" ];
    openssh.authorizedKeys.keys = [];
  };

  users.users.captain = {
    isNormalUser = true;
    description = "captain";
    extraGroups = [ "networkmanager" "wheel" "video" "podman" ];
  };

  users.users.root.openssh.authorizedKeys.keys = [
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDpL+1BEsMSLlWpQuAvDgqNr8ZpVShsUOyjJ+UscbPqK ubuntu@starlite"
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAN2HXpYlT9u2RCJBa05j5pGDh4XK4gBX0sgRwSObvyp opnsense-bog"
  ];

  programs.firefox.enable = true;
  nixpkgs.config.allowUnfree = true;

  environment.systemPackages = with pkgs; [
    git curl wget htop tmux
    python3 python312Packages.pip
    nodejs_22 nodePackages.npm
    gcc cmake gnumake pkg-config
    pciutils usbutils
    ydotool
    llama-cpp
    cudatoolkit
    podman-compose
    distrobox
  ];

  # Container runtime + NVIDIA GPU passthrough for AI lab
  virtualisation.podman = {
    enable = true;
    dockerCompat = true;
    defaultNetwork.settings.dns_enabled = true;
  };
  hardware.nvidia-container-toolkit.enable = true;

  # Llama server
  systemd.services.llama-server = {
    description = "Llama Server (Qwen 3.5 27B)";
    after = [ "network.target" ];
    wantedBy = [ ];  # disabled: vLLM serves Qwen instead
    serviceConfig = {
      Type = "simple";
      Environment = "LD_LIBRARY_PATH=/run/opengl-driver/lib";
      ExecStart = "/opt/llama.cpp/build/bin/llama-server -m /opt/models/qwen3.5-27b-q4km.gguf -ngl 99 -c 32768 --host 0.0.0.0 --port 8080 --reasoning-budget 0 --jinja --parallel 4";
      Restart = "on-failure";
      RestartSec = 5;
    };
  };

  # Qwen proxy (Anthropic API translation)
  systemd.services.qwen-proxy = {
    description = "Qwen Proxy (Anthropic -> OpenAI API)";
    after = [ "llama-server.service" ];
    wantedBy = [ ];  # disabled: vLLM serves Qwen instead
    serviceConfig = {
      Type = "simple";
      ExecStart = "/run/current-system/sw/bin/python3 /opt/qwen-proxy.py 5555";
      Restart = "on-failure";
      RestartSec = 3;
    };
  };


  # Localcode environment
  environment.variables = {
    ANTHROPIC_BASE_URL = "http://127.0.0.1:5555";
    ANTHROPIC_API_KEY = "local";
  };
  environment.shellInit = "export PATH=/opt/bin:\/run/wrappers/bin:/root/.nix-profile/bin:/nix/profile/bin:/root/.local/state/nix/profile/bin:/etc/profiles/per-user/root/bin:/nix/var/nix/profiles/default/bin:/run/current-system/sw/bin";


  # vLLM server (Qwen 3.5 27B with MTP speculative decoding)
  systemd.services.vllm = {
    description = "vLLM OpenAI API Server (Qwen3.5-27B W4A16 + MTP)";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      Environment = [
        "LD_LIBRARY_PATH=/nix/store/1xw5xccqqh1xw3mvd70hyil6x418wxcm-gcc-14.3.0-lib/lib:/run/opengl-driver/lib:/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/lib"
        "HOME=/root"
        "PYTORCH_ALLOC_CONF=expandable_segments:True"
        "CC=/run/current-system/sw/bin/gcc"
        "PATH=/run/current-system/sw/bin:/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/bin"
        "CPATH=/nix/store/qwb5ygz9k8gs5ql9bpxbrsrv12r1icgm-python3-3.13.12/include/python3.13"
      ];
      ExecStart = let
        script = pkgs.writeShellScript "vllm-start" ''
          export LD_LIBRARY_PATH=/nix/store/1xw5xccqqh1xw3mvd70hyil6x418wxcm-gcc-14.3.0-lib/lib:/run/opengl-driver/lib
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:/nix/store/1xw5xccqqh1xw3mvd70hyil6x418wxcm-gcc-14.3.0-lib/lib:/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/lib
          /opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
            --model /opt/models/Qwen3.5-27B-AWQ-textonly \
            --served-model-name qwen3.5-27b \
            --host 0.0.0.0 \
            --port 8001 \
            --max-model-len 4096 \
            --max-num-seqs 4 \
            --gpu-memory-utilization 0.98 \
            --dtype float16 \
            --compilation-config '{"compile_ranges_endpoints": [512]}' \
            --max-num-batched-tokens 1024 \
            --limit-mm-per-prompt '{"image": 0, "video": 0}' \
            --enable-prefix-caching \
            --performance-mode interactivity \
            --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}'
        '';
      in "${script}";
      Restart = "on-failure";
      RestartSec = 10;
    };
  };


  # vLLM watchdog (health check + auto-restart)
  systemd.services.vllm-watchdog = {
    description = "vLLM Health Watchdog";
    after = [ "vllm.service" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      ExecStart = "/run/current-system/sw/bin/bash /opt/vllm-watchdog.sh 8001";
      Restart = "always";
      RestartSec = 10;
    };
  };


  # Cloudflare tunnel for public vLLM API access
  systemd.services.cloudflared-tunnel = {
    description = "Cloudflare Tunnel for vLLM API";
    after = [ "vllm.service" "network-online.target" ];
    wants = [ "network-online.target" ];
    wantedBy = [ "multi-user.target" ];
    serviceConfig = {
      Type = "simple";
      ExecStart = "/opt/bin/cloudflared tunnel --url http://localhost:8001";
      Restart = "always";
      RestartSec = 10;
    };
  };

  system.stateVersion = "25.11";
}
