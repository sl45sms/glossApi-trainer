#!/usr/bin/env python3

import argparse
import select
import socket
import threading


BUFFER_SIZE = 65536


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Proxy a local TCP port to a remote host:port.")
	parser.add_argument("--listen-host", default="127.0.0.1")
	parser.add_argument("--listen-port", type=int, required=True)
	parser.add_argument("--target-host", required=True)
	parser.add_argument("--target-port", type=int, required=True)
	return parser.parse_args()


def relay_bidirectional(client_socket: socket.socket, upstream_socket: socket.socket) -> None:
	open_sockets = [client_socket, upstream_socket]

	while open_sockets:
		try:
			readable, _, exceptional = select.select(open_sockets, [], open_sockets)
		except OSError:
			break

		for sock in exceptional:
			if sock in open_sockets:
				open_sockets.remove(sock)

		for source_socket in readable:
			destination_socket = upstream_socket if source_socket is client_socket else client_socket
			try:
				chunk = source_socket.recv(BUFFER_SIZE)
			except OSError:
				chunk = b""

			if chunk:
				try:
					destination_socket.sendall(chunk)
				except OSError:
					if destination_socket in open_sockets:
						open_sockets.remove(destination_socket)
				continue

			if source_socket in open_sockets:
				open_sockets.remove(source_socket)
			try:
				destination_socket.shutdown(socket.SHUT_WR)
			except OSError:
				pass


def handle_client(client_socket: socket.socket, target_host: str, target_port: int) -> None:
	try:
		upstream_socket = socket.create_connection((target_host, target_port))
	except OSError:
		client_socket.close()
		return

	try:
		relay_bidirectional(client_socket, upstream_socket)
	finally:
		client_socket.close()
		upstream_socket.close()


def main() -> None:
	args = parse_args()
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	server_socket.bind((args.listen_host, args.listen_port))
	server_socket.listen()

	print(
		f"Forwarding {args.listen_host}:{args.listen_port} -> {args.target_host}:{args.target_port}",
		flush=True,
	)

	try:
		while True:
			client_socket, _ = server_socket.accept()
			thread = threading.Thread(
				target=handle_client,
				args=(client_socket, args.target_host, args.target_port),
				daemon=True,
			)
			thread.start()
	finally:
		server_socket.close()


if __name__ == "__main__":
	main()