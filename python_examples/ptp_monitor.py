import os
import socket
import ifaddr
import select


def setup_socket(interface_ip):

    # PTP multicast address
    ptp_multicast_group = "224.0.1.129"
    port = 319  # 319 for event messages, 320 for general messages

    # Create UDP socket
    a_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Allow multiple sockets to use the same PORT number
    a_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind the socket to the specified network interface and PTP port
    try:
        a_socket.bind((interface_ip, port))
    except OSError as e:
        print(f"Error binding socket to interface {interface_ip} and port {port}: {e}")
        exit(1)

    # Join the multicast group on the specified network interface
    try:
        group = socket.inet_aton(ptp_multicast_group)
        mreq = group + socket.inet_aton(interface_ip)
        a_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    except OSError as e:
        print(f"Error joining multicast group: {e}")
        exit(1)

    return a_socket


def monitor_loop(interface_ip):

    try:
        a_socket = setup_socket(interface_ip)
        timeout_s = 1

        while True:
            try:
                r, _, _ = select.select([a_socket], [], [], timeout_s)
                if a_socket in r:
                    data, addr = a_socket.recvfrom(1500)
                    # Process the data as needed
                    # print(f"Received data from {addr}: {data}")
                else:
                    print("No data received, continuing...")
            except KeyboardInterrupt:
                print("Monitoring stopped by user")
                break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Exiting monitor loop.")
        exit(1)


def select_interface(adapter_ips):

    print("Available network interfaces:")
    for i, (ip, name) in enumerate(adapter_ips, 1):
        print(f"{i}: {name} ({ip})")

    while True:
        choice = input("Select a network interface (1-{}): ".format(len(adapter_ips)))
        try:
            choice = int(choice)
            if 1 <= choice <= len(adapter_ips):
                return adapter_ips[choice - 1]
            else:
                print(
                    """Invalid choice.
                      Please select a number between 1 and {}.""".format(
                        len(adapter_ips)
                    )
                )
        except ValueError:
            print("Invalid input. Please enter a number.")


def save_interface_choice(interface_ip, network_name):

    with open("interface_choice.txt", "w") as file:
        file.write(f"{interface_ip},{network_name}")


def load_interface_choice():

    if os.path.exists("interface_choice.txt"):
        with open("interface_choice.txt", "r") as file:
            interface_info = file.readline().strip().split(",")
            if len(interface_info) == 2:
                return interface_info[0], interface_info[1]

    return None, None


if __name__ == "__main__":

    # List available adapters
    adapters = ifaddr.get_adapters()

    # Extract IP addresses and network names from the adapters
    adapter_ips = []
    for adapter in adapters:
        for ip in adapter.ips:
            if isinstance(ip.ip, str):
                adapter_ips.append((ip.ip, adapter.nice_name))

    # Load previously selected network interface choice
    saved_interface_ip, saved_network_name = load_interface_choice()

    if saved_interface_ip and saved_network_name:
        print(
            f"Using saved network interface IP: {saved_interface_ip} ({saved_network_name})"
        )
    else:
        # Select a network interface
        selected_interface_ip, selected_network_name = select_interface(adapter_ips)
        print(
            f"""Selected network interface IP address:
              {selected_interface_ip} ({selected_network_name})"""
        )

        # Save the selected choice
        save_interface_choice(selected_interface_ip, selected_network_name)

        saved_interface_ip, saved_network_name = load_interface_choice()

    # Run the monitor loop indefinitely
    monitor_loop(saved_interface_ip)
