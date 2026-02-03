use crate::errors::DatabaseError;
use ethrex_common::{
    Address, H256, U256,
    types::{AccountState, ChainConfig, Code, CodeMetadata},
};
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

pub mod gen_db;

// Type aliases for cache storage maps
type AccountCache = FxHashMap<Address, AccountState>;
type StorageCache = FxHashMap<(Address, H256), U256>;
type CodeCache = FxHashMap<H256, Code>;

/// Statistics for CachingDatabase lock contention profiling.
/// Enable with ETHREX_PROFILE_CACHE=1 environment variable.
#[derive(Default)]
pub struct CacheStats {
    // Lock acquisition timing (nanoseconds)
    pub read_lock_time_ns: AtomicU64,
    pub write_lock_time_ns: AtomicU64,
    pub read_lock_count: AtomicU64,
    pub write_lock_count: AtomicU64,
    // Cache effectiveness
    pub account_hits: AtomicU64,
    pub account_misses: AtomicU64,
    pub storage_hits: AtomicU64,
    pub storage_misses: AtomicU64,
    pub code_hits: AtomicU64,
    pub code_misses: AtomicU64,
    // Contention detection (read waited > 1µs, likely blocked by write)
    pub read_contentions: AtomicU64,
}

impl CacheStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Print a summary of cache statistics
    pub fn report(&self) {
        let read_count = self.read_lock_count.load(Ordering::Relaxed);
        let write_count = self.write_lock_count.load(Ordering::Relaxed);
        let read_time = self.read_lock_time_ns.load(Ordering::Relaxed);
        let write_time = self.write_lock_time_ns.load(Ordering::Relaxed);
        let contentions = self.read_contentions.load(Ordering::Relaxed);

        let account_hits = self.account_hits.load(Ordering::Relaxed);
        let account_misses = self.account_misses.load(Ordering::Relaxed);
        let storage_hits = self.storage_hits.load(Ordering::Relaxed);
        let storage_misses = self.storage_misses.load(Ordering::Relaxed);
        let code_hits = self.code_hits.load(Ordering::Relaxed);
        let code_misses = self.code_misses.load(Ordering::Relaxed);

        let avg_read_ns = if read_count > 0 {
            read_time / read_count
        } else {
            0
        };
        let avg_write_ns = if write_count > 0 {
            write_time / write_count
        } else {
            0
        };

        let account_total = account_hits + account_misses;
        let storage_total = storage_hits + storage_misses;
        let code_total = code_hits + code_misses;

        let account_hit_rate = if account_total > 0 {
            (account_hits as f64 / account_total as f64) * 100.0
        } else {
            0.0
        };
        let storage_hit_rate = if storage_total > 0 {
            (storage_hits as f64 / storage_total as f64) * 100.0
        } else {
            0.0
        };
        let code_hit_rate = if code_total > 0 {
            (code_hits as f64 / code_total as f64) * 100.0
        } else {
            0.0
        };

        let contention_rate = if read_count > 0 {
            (contentions as f64 / read_count as f64) * 100.0
        } else {
            0.0
        };

        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("📊 CachingDatabase Lock Contention Profile");
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!();
        eprintln!("Lock Acquisition:");
        eprintln!("  Read locks:  {:>10} calls, {:>6} ns avg", read_count, avg_read_ns);
        eprintln!("  Write locks: {:>10} calls, {:>6} ns avg", write_count, avg_write_ns);
        eprintln!("  Contentions: {:>10} ({:.2}% of reads waited >1µs)", contentions, contention_rate);
        eprintln!();
        eprintln!("Cache Effectiveness:");
        eprintln!("  Accounts: {:>8} hits, {:>8} misses ({:.1}% hit rate)", account_hits, account_misses, account_hit_rate);
        eprintln!("  Storage:  {:>8} hits, {:>8} misses ({:.1}% hit rate)", storage_hits, storage_misses, storage_hit_rate);
        eprintln!("  Code:     {:>8} hits, {:>8} misses ({:.1}% hit rate)", code_hits, code_misses, code_hit_rate);
        eprintln!();
        eprintln!("DashMap Recommendation:");
        if contention_rate > 5.0 {
            eprintln!("  ⚠️  HIGH contention ({:.1}%) - DashMap likely beneficial", contention_rate);
        } else if contention_rate > 1.0 {
            eprintln!("  ⚡ MODERATE contention ({:.1}%) - DashMap may help", contention_rate);
        } else {
            eprintln!("  ✅ LOW contention ({:.1}%) - RwLock is fine", contention_rate);
        }
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

fn should_profile() -> bool {
    std::env::var("ETHREX_PROFILE_CACHE").map_or(false, |v| v == "1" || v == "true")
}

pub trait Database: Send + Sync {
    fn get_account_state(&self, address: Address) -> Result<AccountState, DatabaseError>;
    fn get_storage_value(&self, address: Address, key: H256) -> Result<U256, DatabaseError>;
    fn get_block_hash(&self, block_number: u64) -> Result<H256, DatabaseError>;
    fn get_chain_config(&self) -> Result<ChainConfig, DatabaseError>;
    fn get_account_code(&self, code_hash: H256) -> Result<Code, DatabaseError>;
    fn get_code_metadata(&self, code_hash: H256) -> Result<CodeMetadata, DatabaseError>;
}

/// A database wrapper that caches state lookups for parallel pre-warming.
///
/// This enables parallel warming workers to share cached data, and allows
/// the sequential execution phase to reuse warmed state. Reduces redundant
/// database/trie lookups when multiple transactions touch the same accounts.
///
/// Thread-safe via RwLock - optimized for read-heavy concurrent access.
///
/// This caching database is inspired by reth's overlay/proof worker cache.
///
/// # Profiling
///
/// Set `ETHREX_PROFILE_CACHE=1` to enable lock contention profiling.
/// Call `stats()` to get statistics, or `report_stats()` to print a summary.
pub struct CachingDatabase {
    inner: Arc<dyn Database>,
    /// Cached account states (balance, nonce, code_hash, storage_root)
    accounts: RwLock<AccountCache>,
    /// Cached storage values
    storage: RwLock<StorageCache>,
    /// Cached contract code
    code: RwLock<CodeCache>,
    /// Lock contention statistics (only collected when ETHREX_PROFILE_CACHE=1)
    stats: CacheStats,
    /// Whether profiling is enabled
    profile: bool,
}

impl CachingDatabase {
    pub fn new(inner: Arc<dyn Database>) -> Self {
        let profile = should_profile();
        if profile {
            eprintln!("[CachingDatabase] Profiling enabled (ETHREX_PROFILE_CACHE=1)");
        }
        Self {
            inner,
            accounts: RwLock::new(FxHashMap::default()),
            storage: RwLock::new(FxHashMap::default()),
            code: RwLock::new(FxHashMap::default()),
            stats: CacheStats::new(),
            profile,
        }
    }

    /// Get a reference to the cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Print a summary of cache statistics to stderr
    pub fn report_stats(&self) {
        self.stats.report();
    }

    fn read_accounts(&self) -> Result<RwLockReadGuard<'_, AccountCache>, DatabaseError> {
        if self.profile {
            let start = Instant::now();
            let guard = self.accounts.read().map_err(poison_error_to_db_error)?;
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            self.stats.read_lock_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
            self.stats.read_lock_count.fetch_add(1, Ordering::Relaxed);
            if elapsed_ns > 1000 {
                // > 1µs indicates likely contention
                self.stats.read_contentions.fetch_add(1, Ordering::Relaxed);
            }
            Ok(guard)
        } else {
            self.accounts.read().map_err(poison_error_to_db_error)
        }
    }

    fn write_accounts(&self) -> Result<RwLockWriteGuard<'_, AccountCache>, DatabaseError> {
        if self.profile {
            let start = Instant::now();
            let guard = self.accounts.write().map_err(poison_error_to_db_error)?;
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            self.stats.write_lock_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
            self.stats.write_lock_count.fetch_add(1, Ordering::Relaxed);
            Ok(guard)
        } else {
            self.accounts.write().map_err(poison_error_to_db_error)
        }
    }

    fn read_storage(&self) -> Result<RwLockReadGuard<'_, StorageCache>, DatabaseError> {
        if self.profile {
            let start = Instant::now();
            let guard = self.storage.read().map_err(poison_error_to_db_error)?;
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            self.stats.read_lock_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
            self.stats.read_lock_count.fetch_add(1, Ordering::Relaxed);
            if elapsed_ns > 1000 {
                self.stats.read_contentions.fetch_add(1, Ordering::Relaxed);
            }
            Ok(guard)
        } else {
            self.storage.read().map_err(poison_error_to_db_error)
        }
    }

    fn write_storage(&self) -> Result<RwLockWriteGuard<'_, StorageCache>, DatabaseError> {
        if self.profile {
            let start = Instant::now();
            let guard = self.storage.write().map_err(poison_error_to_db_error)?;
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            self.stats.write_lock_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
            self.stats.write_lock_count.fetch_add(1, Ordering::Relaxed);
            Ok(guard)
        } else {
            self.storage.write().map_err(poison_error_to_db_error)
        }
    }

    fn read_code(&self) -> Result<RwLockReadGuard<'_, CodeCache>, DatabaseError> {
        if self.profile {
            let start = Instant::now();
            let guard = self.code.read().map_err(poison_error_to_db_error)?;
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            self.stats.read_lock_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
            self.stats.read_lock_count.fetch_add(1, Ordering::Relaxed);
            if elapsed_ns > 1000 {
                self.stats.read_contentions.fetch_add(1, Ordering::Relaxed);
            }
            Ok(guard)
        } else {
            self.code.read().map_err(poison_error_to_db_error)
        }
    }

    fn write_code(&self) -> Result<RwLockWriteGuard<'_, CodeCache>, DatabaseError> {
        if self.profile {
            let start = Instant::now();
            let guard = self.code.write().map_err(poison_error_to_db_error)?;
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            self.stats.write_lock_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
            self.stats.write_lock_count.fetch_add(1, Ordering::Relaxed);
            Ok(guard)
        } else {
            self.code.write().map_err(poison_error_to_db_error)
        }
    }
}

fn poison_error_to_db_error<T>(err: PoisonError<T>) -> DatabaseError {
    DatabaseError::Custom(format!("Cache lock poisoned: {err}"))
}

impl Database for CachingDatabase {
    fn get_account_state(&self, address: Address) -> Result<AccountState, DatabaseError> {
        // Check cache first
        if let Some(state) = self.read_accounts()?.get(&address).copied() {
            if self.profile {
                self.stats.account_hits.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(state);
        }

        if self.profile {
            self.stats.account_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Cache miss: query underlying database
        let state = self.inner.get_account_state(address)?;

        // Populate cache (AccountState is Copy, no clone needed)
        self.write_accounts()?.insert(address, state);

        Ok(state)
    }

    fn get_storage_value(&self, address: Address, key: H256) -> Result<U256, DatabaseError> {
        // Check cache first
        if let Some(value) = self.read_storage()?.get(&(address, key)).copied() {
            if self.profile {
                self.stats.storage_hits.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(value);
        }

        if self.profile {
            self.stats.storage_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Cache miss: query underlying database
        let value = self.inner.get_storage_value(address, key)?;

        // Populate cache (U256 is Copy, no clone needed)
        self.write_storage()?.insert((address, key), value);

        Ok(value)
    }

    fn get_block_hash(&self, block_number: u64) -> Result<H256, DatabaseError> {
        // Block hashes don't benefit much from caching here
        // (they're already cached in StoreVmDatabase)
        self.inner.get_block_hash(block_number)
    }

    fn get_chain_config(&self) -> Result<ChainConfig, DatabaseError> {
        // Chain config is constant, no need to cache
        self.inner.get_chain_config()
    }

    fn get_account_code(&self, code_hash: H256) -> Result<Code, DatabaseError> {
        // Check cache first
        if let Some(code) = self.read_code()?.get(&code_hash).cloned() {
            if self.profile {
                self.stats.code_hits.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(code);
        }

        if self.profile {
            self.stats.code_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Cache miss: query underlying database
        let code = self.inner.get_account_code(code_hash)?;

        // Populate cache (Code contains Bytes which is ref-counted, clone is cheap)
        self.write_code()?.insert(code_hash, code.clone());

        Ok(code)
    }

    fn get_code_metadata(&self, code_hash: H256) -> Result<CodeMetadata, DatabaseError> {
        // Delegate directly to the underlying database.
        // The underlying Store already has its own code_metadata_cache,
        // so we don't need to duplicate caching here.
        self.inner.get_code_metadata(code_hash)
    }
}

impl Drop for CachingDatabase {
    fn drop(&mut self) {
        if self.profile {
            self.stats.report();
        }
    }
}
