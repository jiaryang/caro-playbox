#!/bin/bash

# Check NUMA balancing
if [ -r /proc/sys/kernel/numa_balancing ]; then
    numa_balancing=$(cat /proc/sys/kernel/numa_balancing)
    if [ "$numa_balancing" -eq 0 ]; then
        echo "NUMA auto-balancing is disabled. This is the recommended setting."
    else
        echo "NUMA auto-balancing is enabled (value: $numa_balancing). It is recommended to disable it for performance."
        echo "Suggested fix: sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'"
    fi
else
    echo "Cannot read /proc/sys/kernel/numa_balancing without sudo. Please run with sudo for accurate results."
fi

# Check iommu=pt in GRUB config
if [ -r /etc/default/grub ]; then
    grub_config=$(cat /etc/default/grub)
    if [[ "$grub_config" == *iommu=pt* ]]; then
        echo "iommu=pt is set in GRUB config. This is the recommended setting."
        grep "iommu=pt" /etc/default/grub
    else
        echo "iommu=pt is NOT set in GRUB config. It is recommended to enable it for performance."
    fi
else
    echo "Cannot read /etc/default/grub without sudo. Please run with sudo for accurate results."
fi
