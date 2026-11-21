// respeaker: read or set the ReSpeaker 4 Mic Array's XVF-3000 parameters (go/pkg/respeaker).
//
//	respeaker                 every parameter with its value
//	respeaker NAME            one
//	respeaker NAME VALUE ...  set (NAME VALUE pairs), each read back
package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/ibnHatab/fsttm/go/pkg/respeaker"
)

func main() {
	d, err := respeaker.Open()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer d.Close()
	a := os.Args[1:]
	switch {
	case len(a) == 0:
		for _, n := range respeaker.Names() {
			v, err := d.Read(n)
			if err != nil {
				fmt.Printf("%-22s ERROR %v\n", n, err)
				continue
			}
			fmt.Printf("%-22s %-10.4g %s\n", n, v, respeaker.Params[n].What)
		}
	case len(a) == 1:
		v, err := d.Read(a[0])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("%s %.6g\n", a[0], v)
	case len(a)%2 == 0:
		set := map[string]float64{}
		for i := 0; i < len(a); i += 2 {
			v, err := strconv.ParseFloat(a[i+1], 64)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s: %v\n", a[i], err)
				os.Exit(2)
			}
			set[a[i]] = v
		}
		lines, err := d.Apply(set)
		for _, l := range lines {
			fmt.Println(l)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	default:
		fmt.Fprintln(os.Stderr, "respeaker [NAME [VALUE] ...]")
		os.Exit(2)
	}
}
