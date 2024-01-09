#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <windows.h>
#include <winnt.h>

#include <string>
#include <vector>
#include <Psapi.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <iterator>
#include <array>

#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "dbghelp.lib")

// Some versions of imagehlp.dll lack the proper packing directives themselves
// so we need to do it.
#pragma pack( push, before_imagehlp, 8 )
#include <imagehlp.h>
#pragma pack( pop, before_imagehlp )

struct module_data {
    std::string image_name;
    std::string module_name;
    void *base_address;
    DWORD load_size;
};
typedef std::vector<module_data> ModuleList;

HANDLE thread_ready;

bool show_stack(std::ostream &, HANDLE hThread, CONTEXT& c);
DWORD Filter( EXCEPTION_POINTERS *ep );
void *load_modules_symbols( HANDLE hProcess, DWORD pid );


// if you use C++ exception handling: install a translator function
// with set_se_translator(). In the context of that function (but *not*
// afterwards), you can either do your stack dump, or save the CONTEXT
// record as a local copy. Note that you must do the stack dump at the
// earliest opportunity, to avoid the interesting stack-frames being gone
// by the time you do the dump.
DWORD Filter(EXCEPTION_POINTERS *ep) {
    HANDLE thread;

    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),
        GetCurrentProcess(), &thread, 0, false, DUPLICATE_SAME_ACCESS);
    std::cout << "Walking stack.";
    show_stack(std::cout, thread, *(ep->ContextRecord));
    std::cout << "\nEnd of stack walk.\n";
    CloseHandle(thread);

    return EXCEPTION_EXECUTE_HANDLER;
}

class SymHandler { 
    HANDLE p;
public:
    SymHandler(HANDLE process, char const *path=NULL, bool intrude = false) : p(process) { 
        if (!SymInitialize(p, path, intrude)) 
            throw(std::logic_error("Unable to initialize symbol handler"));
    }
    ~SymHandler() { SymCleanup(p); }
};

#ifdef _M_X64
STACKFRAME64 init_stack_frame(CONTEXT c) {
    STACKFRAME64 s;
    s.AddrPC.Offset = c.Rip;
    s.AddrPC.Mode = AddrModeFlat;
    s.AddrStack.Offset = c.Rsp;
    s.AddrStack.Mode = AddrModeFlat;    
    s.AddrFrame.Offset = c.Rbp;
    s.AddrFrame.Mode = AddrModeFlat;
    return s;
}
#else
STACKFRAME64 init_stack_frame(CONTEXT c) {
    STACKFRAME64 s;
    s.AddrPC.Offset = c.Eip;
    s.AddrPC.Mode = AddrModeFlat;
    s.AddrStack.Offset = c.Esp;
    s.AddrStack.Mode = AddrModeFlat;    
    s.AddrFrame.Offset = c.Ebp;
    s.AddrFrame.Mode = AddrModeFlat;
    return s;
}
#endif

void sym_options(DWORD add, DWORD remove=0) {
    DWORD symOptions = SymGetOptions();
    symOptions |= add;
    symOptions &= ~remove;
    SymSetOptions(symOptions);
}

//Returns the last Win32 error, in string format. Returns an empty string if there is no error.
std::string GetLastErrorAsString()
{
    //Get the error message ID, if any.
    DWORD errorMessageID = ::GetLastError();
    if(errorMessageID == 0) {
        return std::string(); //No error message has been recorded
    }
    
    LPSTR messageBuffer = nullptr;

    //Ask Win32 to give us the string version of that message ID.
    //The parameters we pass in, tell Win32 to create the buffer that holds the message for us (because we don't yet know how long the message string will be).
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
    
    //Copy the error message into a std::string.
    std::string message(messageBuffer, size);
    
    //Free the Win32's string's buffer.
    LocalFree(messageBuffer);
            
    return message;
}

class symbol { 
    typedef IMAGEHLP_SYMBOL64 sym_type;
    sym_type *sym;
    static const int max_name_len = 1024;
    bool bad_sym = false;
public:
    symbol(HANDLE process, DWORD64 address) : sym((sym_type *)::operator new(sizeof(*sym) + max_name_len)) {
        memset(sym, '\0', sizeof(*sym) + max_name_len);
        sym->SizeOfStruct = sizeof(*sym);
        sym->MaxNameLength = max_name_len;
        DWORD64 displacement;

        if(!SymGetSymFromAddr64(process, address, &displacement, sym)) {
            bad_sym = true;
        }
    }

    std::string name() { return bad_sym ? "bad_sym" : std::string(sym->Name); }
    std::string undecorated_name() { 
        if(bad_sym) return "bad_sym";
        std::vector<char> und_name(max_name_len);
        UnDecorateSymbolName(sym->Name, &und_name[0], max_name_len, UNDNAME_COMPLETE);
        return std::string(&und_name[0], strlen(&und_name[0]));
    }
};

bool show_stack(std::ostream &os, HANDLE hThread, CONTEXT& c) {
    HANDLE process = GetCurrentProcess();
    int frame_number=0;
    DWORD offset_from_symbol=0;
    IMAGEHLP_LINE64 line = {0};

    SymHandler handler(process, NULL, true);

    sym_options(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);

    void *base = load_modules_symbols(process, GetCurrentProcessId());

    STACKFRAME64 s = init_stack_frame(c);

    line.SizeOfStruct = sizeof line;

    IMAGE_NT_HEADERS *h = ImageNtHeader(base);
    DWORD image_type = h->FileHeader.Machine;

    do {
        if (!StackWalk64(image_type, process, hThread, &s, &c, NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL))
            return false;

        os << std::setw(3) << "\n" << frame_number << "\t";
        if ( s.AddrPC.Offset != 0 ) {
            std::cout << symbol(process, s.AddrPC.Offset).undecorated_name();

            if (SymGetLineFromAddr64( process, s.AddrPC.Offset, &offset_from_symbol, &line ) ) 
                    os << "\t" << line.FileName << "(" << line.LineNumber << ")";
        }
        else
            os << "(No Symbols: PC == 0)";
        ++frame_number;
    } while (s.AddrReturn.Offset != 0);
    return true;
}

class get_mod_info {
    HANDLE process;
    static const int buffer_length = 4096;
public:
    get_mod_info(HANDLE h) : process(h) {}

    module_data operator()(HMODULE module) { 
        module_data ret;
        char temp[buffer_length];
        MODULEINFO mi;

        GetModuleInformation(process, module, &mi, sizeof(mi));
        ret.base_address = mi.lpBaseOfDll;
        ret.load_size = mi.SizeOfImage;

        GetModuleFileNameEx(process, module, temp, sizeof(temp));
        ret.image_name = temp;
        GetModuleBaseName(process, module, temp, sizeof(temp));
        ret.module_name = temp;
        std::vector<char> img(ret.image_name.begin(), ret.image_name.end());
        std::vector<char> mod(ret.module_name.begin(), ret.module_name.end());
        SymLoadModule64(process, 0, &img[0], &mod[0], (DWORD64)ret.base_address, ret.load_size);
        return ret;
    }
};

void *load_modules_symbols(HANDLE process, DWORD pid) {
    ModuleList modules;

    DWORD cbNeeded;
    std::vector<HMODULE> module_handles(1);

    EnumProcessModules(process, &module_handles[0], module_handles.size() * sizeof(HMODULE), &cbNeeded);
    module_handles.resize(cbNeeded/sizeof(HMODULE));
    EnumProcessModules(process, &module_handles[0], module_handles.size() * sizeof(HMODULE), &cbNeeded);

    std::transform(module_handles.begin(), module_handles.end(), std::back_inserter(modules), get_mod_info(process));
    return modules[0].base_address;
}

std::optional<std::string> libenvpp_convert_string(const std::wstring& str)
{
    printf("libenvpp_convert_string " SIMSYCL_LINE_STRING "\n");
	const auto buffer_size =
	    WideCharToMultiByte(CP_UTF8, 0, str.c_str(), static_cast<int>(str.length()), nullptr, 0, nullptr, nullptr);
    printf("libenvpp_convert_string " SIMSYCL_LINE_STRING "\n");
    std::wcout << "     buffer_size " << buffer_size << std::endl;
	if (buffer_size == 0) {
		return {};
	}
	auto buffer = std::string(buffer_size, '\0');
    printf("libenvpp_convert_string " SIMSYCL_LINE_STRING "\n");
    std::wcout << "   lpWideCharStr " << str.c_str() << std::endl;
    std::wcout << "     cchWideChar " << str.length() << std::endl;
    std::cout <<  "  lpMultiByteStr " << buffer << std::endl;
    std::wcout << "     cchWideChar " << str.length() << std::endl;
	[[maybe_unused]] const auto res = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), static_cast<int>(str.length()),
	                                                      buffer.data(), buffer_size, nullptr, nullptr);
    printf("libenvpp_convert_string " SIMSYCL_LINE_STRING "\n");
	assert(res == buffer_size);
	return buffer;
}

std::optional<std::wstring> libenvpp_convert_string(const std::string& str)
{
    printf("libenvpp_convert_string " SIMSYCL_LINE_STRING "\n");
	const auto buffer_size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.length()), nullptr, 0);
	if (buffer_size == 0) {
		return {};
	}
    printf("libenvpp_convert_string " SIMSYCL_LINE_STRING "\n");
	auto buffer = std::wstring(buffer_size, L'\0');
	[[maybe_unused]] const auto res =
	    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), static_cast<int>(str.length()), buffer.data(), buffer_size);
    printf("libenvpp_convert_string " SIMSYCL_LINE_STRING "\n");
	assert(res == buffer_size);
	return buffer;
}

[[nodiscard]] std::unordered_map<std::string, std::string> libenvpp_get_environment()
{
    printf("libenvpp_get_environment " SIMSYCL_LINE_STRING "\n");
	auto env_map = std::unordered_map<std::string, std::string>{};

	const auto environment = GetEnvironmentStringsW();
	if (!environment) {
		return env_map;
	}
    printf("libenvpp_get_environment " SIMSYCL_LINE_STRING "\n");

	for (const auto* var = environment; *var; ++var) {
        printf("libenvpp_get_environment " SIMSYCL_LINE_STRING "\n");
		auto var_name_value = std::array<std::wstring, 2>{};
		auto idx = std::size_t{0};
		for (; *var; ++var) {
			if (idx == 0 && *var == L'=') {
				++idx;
			} else {
				var_name_value[idx] += *var;
			}
		}
        printf("libenvpp_get_environment " SIMSYCL_LINE_STRING "\n");
		if (!var_name_value[0].empty()) {
            auto key = libenvpp_convert_string(var_name_value[0]);
            auto value = libenvpp_convert_string(var_name_value[1]);
            if(key && value) {
				env_map[*key] = *value;
            }
		}
        printf("libenvpp_get_environment " SIMSYCL_LINE_STRING "\n");
	}

	[[maybe_unused]] const auto env_strings_were_freed = FreeEnvironmentStringsW(environment);
	assert(env_strings_were_freed);

	return env_map;
}


using namespace sycl;

SIMSYCL_START_IGNORING_DEPRECATIONS

TEST_CASE("Calls to the deprecated parallel_for signature are not ambiguous", "[ambiguity][parallel_for]") {
    // spawn a thread which prints the backtrace of the current thread after 10 seconds
    auto this_tread = GetCurrentThread();
    std::atomic<bool> stopped = false;
    auto t = std::thread([&]() {
		Sleep(200000);
        if(!stopped) {
            CONTEXT c;
            memset(&c, 0, sizeof(CONTEXT));
            c.ContextFlags = CONTEXT_FULL;
            GetThreadContext(this_tread, &c);
            show_stack(std::cout, this_tread, c);
            exit(0);
        }
    });

    auto env = libenvpp_get_environment();
    for(auto& [k, v] : env) {
		printf("%s=%s\n", k.c_str(), v.c_str());
	}
    printf("\n\n");

    printf("START\n");
    queue q;
    printf("q\n");
    constexpr int offset = 7;
    printf("1D\n");
    SECTION("1D") {
        printf("1D A\n");
        q.submit([&](handler &cgh) {
            printf("1D B\n");
            cgh.parallel_for(range<1>{1}, id<1>{offset}, [=](id<1> i) {
                printf("1D C\n");
                CHECK(i[0] == offset);
                printf("1D D\n");
            });
            printf("1D E\n");
        });
        printf("1D F\n");
    }
    printf("2D\n");
    SECTION("2D") {
        printf("2D A\n");
        q.submit([&](handler &cgh) {
            printf("2D B\n");
            cgh.parallel_for(range<2>{1, 1}, id<2>{0, offset}, [=](id<2> i) {
                printf("2D C\n");
                CHECK(i == id<2>{0, offset});
                printf("2D D\n");
            });
            printf("2D F\n");
        });
    }
    printf("3D\n");
    SECTION("3D") {
        q.submit([&](handler &cgh) {
            cgh.parallel_for(range<3>{1, 1, 1}, id<3>{0, offset, 0}, [=](id<3> i) { CHECK(i == id<3>{0, offset, 0}); });
        });
    }
    printf("END\n");

    stopped = true;
    t.join();
}

SIMSYCL_STOP_IGNORING_DEPRECATIONS
